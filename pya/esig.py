"""This module is used to create editable audio signals.

This module is used to create editable audio signals, which can be manipulated in time and pitch.
The editing is non-destructive,
meaning that the original audio signal is not modified and changes can be undone.
"""

from typing import Type
from abc import ABC, abstractmethod
import json
import binascii
from amfm_decompy import basic_tools, pYAAPT
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pytsmod as tsm
import scipy.ndimage
import scipy.io.wavfile
from pya.asig import Asig


class Esig:
    """The main class for editable audio signals.

    This class represents an editable audio signal, which can be manipulated in time and pitch.
    To allow non-destructive editing, the unmodified audio signal is stored as an Asig object.
    """

    def __init__(
        self,
        obj_input: any,
        algorithm: str = "yaapt",
        max_vibrato_extent: float = 40,
        max_vibrato_inaccuracy: float = 0.5,
        min_event_length: float = 0.1,
    ) -> None:
        """Creates a new editable audio signal from an existing audio signal.

        Parameters
        ----------
        obj_input : any
            Either the json string of an Esig object, or an Asig object.
        algorithm : str
            The algorithm to be used to guess the pitch of the audio signal.
            Possible values are: 'yaapt'
        max_vibrato_extent : float
            The maximum difference between the average pitch of a event to each pitch in the event,
            in cents (100 cents = 1 semitone).
            Voice vibrato is usually below 100 cents.
        max_vibrato_inaccuracy : float
            A factor (between 0 and 1) that determines how accurate
            the pitch of a event has to be within the event, when the signal has vibrato.
            A value near 0 means that the pitch has to be very accurate,
            e.g. the vibrato has to be very even.
        min_event_length : float
            The minimum length of a event in seconds.
            Events shorter than this will be filtered out.
        """

        if isinstance(obj_input, str):
            json_dict = json.loads(obj_input)

            # Load the signal from the json dict
            signal = np.frombuffer(
                binascii.a2b_base64(json_dict["signal_base64"]), dtype=np.float32
            )
            num_channels = json_dict["num_channels"]
            if num_channels != 1:
                signal = signal.reshape(
                    (
                        int(len(signal) / num_channels),
                        num_channels,
                    )
                )  # The loaded array is flattened, we need to reshape it
            sample_rate = json_dict["sample_rate"]
            self.asig = Asig(signal, sample_rate)
            self.algorithm = json_dict["algorithm"]
            self.max_vibrato_extent = json_dict["max_vibrato_extent"]
            self.max_vibrato_inaccuracy = json_dict["max_vibrato_inaccuracy"]
            self.min_event_length = json_dict["min_event_length"]

            # Load the edits from the json dict
            self.edits = []
            for edit_json in json_dict["edits"]:
                if edit_json["type"] == "pitch_change":
                    self.edits.append(
                        PitchChange(
                            edit_json["start"],
                            edit_json["end"],
                            edit_json["change"],
                            edit_json["algorithm"],
                        )
                    )
                elif edit_json["type"] == "length_change":
                    self.edits.append(
                        LengthChange(
                            edit_json["start"],
                            edit_json["end"],
                            edit_json["stretch_factor"],
                            edit_json["algorithm"],
                        )
                    )
                elif edit_json["type"] == "event_modification":
                    self.edits.append(
                        EventModification(
                            edit_json["start"],
                            edit_json["end"],
                            edit_json["new_start"],
                            edit_json["new_end"],
                            edit_json["offset"],
                        )
                    )
                elif edit_json["type"] == "event_creation":
                    self.edits.append(
                        EventCreation(
                            edit_json["start"],
                            edit_json["end"],
                        )
                    )
                elif edit_json["type"] == "event_deletion":
                    self.edits.append(
                        EventDeletion(
                            edit_json["start"],
                            edit_json["end"],
                        )
                    )
                elif edit_json["type"] == "pitch_curve_edit":
                    target_pitch = np.frombuffer(
                        binascii.a2b_base64(edit_json["target_pitch"]),
                        dtype=np.float32,
                    )
                    target_pitch = target_pitch.reshape(
                        (
                            int(len(target_pitch) / 2),
                            2,
                        )
                    )
                    self.edits.append(
                        PitchCurveEdit(
                            edit_json["start"],
                            edit_json["end"],
                            target_pitch,
                            edit_json["algorithm"],
                        )
                    )
                else:
                    raise ValueError("Invalid edit type")

            self.cache = Cache(self)
            self.cache.reapply()  # Apply all edits to the cache
        elif isinstance(obj_input, Asig):
            self.asig = obj_input
            self.algorithm = algorithm
            self.max_vibrato_extent = max_vibrato_extent
            self.max_vibrato_inaccuracy = max_vibrato_inaccuracy
            self.min_event_length = min_event_length
            self.edits = []

            self.cache = Cache(
                self
            )  # Initialize the cache, storing the results of edits
        else:
            raise ValueError("Invalid input")

    def undo_last(self) -> None:
        """Undos the last edit on this signal."""

        self.edits.pop()
        self.cache.reapply()  # We need to reapply all edits in the cache

    def change_pitch(
        self,
        start: float,
        end: float,
        semitones: float,
        algorithm: str = "tdpsola",
    ) -> None:
        """Changes the pitch of the given sample range by the given amount.

        Parameters
        ----------
        start : float
            The starting second to change (inclusive)
        end : float
            The ending second to change (exclusive)
        semitones : float
            The amount of semitones to change the pitch with.
            0.0 means no change.
        algorithm : str, optional
            The algorithm to change the pitch with, by default "tdpsola".
            Currently, only "tdpsola" is supported.
        """

        # Convert seconds to samples
        start = int(start * self.asig.sr)
        end = int(end * self.asig.sr)

        self.edits.append(PitchChange(start, end, semitones, algorithm))
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

    def change_event_pitch(
        self, event_index: int, semitones: float, algorithm: str = "tdpsola"
    ) -> None:
        """Changes the pitch of the given event by the given amount.

        Parameters
        ----------
        event_index : int
            The index of the event to change. (Can be found with print_events())
        semitones : float
            The amount of semitones to change the pitch with.
            0.0 means no change.
        algorithm : str, optional
            The algorithm to change the pitch with, by default "tdpsola".
            Currently, only "tdpsola" is supported.
        """

        # Get the event
        event = self.cache.events[event_index]

        # Convert samples to seconds
        start = event.start / self.asig.sr
        end = event.end / self.asig.sr

        self.change_pitch(start, end, semitones, algorithm)

    def change_length(
        self, start: float, end: float, stretch_factor: float, algorithm: str = "wsola"
    ) -> None:
        """Changes the length of the given sample range by the given amount.

        Parameters
        ----------
        start : float
            The starting second to change (inclusive)
        end : float
            The ending second to change (exclusive)
        stretch_factor : float
            The factor to change the length with.
            1.0 means no change.
        algorithm : str, optional
            The algorithm to change the length with, by default "wsola".
            Currently, only "wsola" is supported.
        """

        # Convert seconds to samples
        start = int(start * self.asig.sr)
        end = int(end * self.asig.sr)

        self.edits.append(LengthChange(start, end, stretch_factor, algorithm))
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

    def change_event_length(
        self, event_index: int, stretch_factor: float, algorithm: str = "wsola"
    ) -> None:
        """Changes the length of the given event by the given amount.

        Parameters
        ----------
        event_index : int
            The index of the event to change. (Can be found with print_events())
        stretch_factor : float
            The factor to change the length with.
            1.0 means no change.
        algorithm : str, optional
            The algorithm to change the length with, by default "wsola".
            Currently, only "wsola" is supported.
        """

        # Get the event
        event = self.cache.events[event_index]

        # Convert samples to seconds
        start = event.start / self.asig.sr
        end = event.end / self.asig.sr

        self.change_length(start, end, stretch_factor, algorithm)

    def change_pitch_curve(
        self,
        start: float,
        end: float,
        pitch_curve: np.ndarray,
        algorithm: str = "tdpsola",
    ) -> None:
        """Changes the pitch of the given sample range by the given pitch curve.

        Parameters
        ----------
        start : float
            The starting second to change (inclusive)
        end : float
            The ending second to change (exclusive)
        pitch_curve : np.ndarray
            The target pitch curve, given as a numpy array of tuples (time, pitch).
            The time starts at 0 (start of the edit) and ends at 1 (end of the edit),
            given as a fraction of the length of the edit.
            The pitch is given in semitones as the difference to the original pitch.
        algorithm : str, optional
            The algorithm to change the pitch with, by default "tdpsola".
            Currently, only "tdpsola" is supported.
        """

        # Convert seconds to samples
        start = int(start * self.asig.sr)
        end = int(end * self.asig.sr)

        self.edits.append(PitchCurveEdit(start, end, pitch_curve, algorithm))
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

    def change_event_pitch_curve(
        self, event_index: int, pitch_curve: np.ndarray, algorithm: str = "tdpsola"
    ) -> None:
        """Changes the pitch of the given event by the given pitch curve.

        Parameters
        ----------
        event_index : int
            The index of the event to change. (Can be found with print_events())
        pitch_curve : np.ndarray
            The target pitch curve, given as a numpy array of tuples (time, pitch).
            The time starts at 0 (start of the edit) and ends at 1 (end of the edit),
            given as a fraction of the length of the edit.
        algorithm : str, optional
            The algorithm to change the pitch with, by default "tdpsola".
            Currently, only "tdpsola" is supported.
        """

        # Get the event
        event = self.cache.events[event_index]

        # Convert samples to seconds
        start = event.start / self.asig.sr
        end = event.end / self.asig.sr

        self.change_pitch_curve(start, end, pitch_curve, algorithm)

    def modify_event(
        self,
        event_index: int,
        new_start: float | None,
        new_end: float | None,
        offset: float | None,
    ) -> None:
        """Manually modifies an auto detected event.

        Parameters
        ----------
        event_index : int
            The index of the event to change. (Can be found with print_events())
        new_start : float | None
            The new starting second of the event.
            None means no change.
        new_end : float | None
            The new ending second of the event.
            None means no change.
        offset : float | None
            The offset to apply to the event, after the new boundaries are applied.
            0 or None means no offset.
        """

        # Get the event
        event = self.cache.events[event_index]

        # Convert seconds to samples
        if new_start is not None:
            new_start = int(new_start * self.asig.sr)
        else:
            new_start = event.start
        if new_end is not None:
            new_end = int(new_end * self.asig.sr)
        else:
            new_end = event.end
        if offset is not None:
            offset = int(offset * self.asig.sr)
        else:
            offset = 0

        self.edits.append(
            EventModification(event.start, event.end, new_start, new_end, offset)
        )
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

    def create_event(self, start: float, end: float) -> None:
        """Creates an event with the given time range.

        Parameters
        ----------
        start : float
            The starting second of the event.
        end : float
            The ending second of the event.
        """

        # Convert seconds to samples
        start = int(start * self.asig.sr)
        end = int(end * self.asig.sr)

        self.edits.append(EventCreation(start, end))
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

    def delete_event(self, event_index: int) -> None:
        """Deletes the given event.
        If multiple events exist in the given range of the selected event,
        all of them will be deleted.

        Parameters
        ----------
        event_index : int
            The index of the event to delete. (Can be found with print_events())
        """

        # Get the event
        event = self.cache.events[event_index]

        self.edits.append(EventDeletion(event.start, event.end))
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

    def _avg_pitch(self, event: Type["Event"]) -> float:
        """Calculates the average pitch of an event on the current pitch in cache.
        Uses interpolation for better accuracy when converting from signal samples to pitch samples.

        Parameters
        ----------
        event : Type[&quot;Event&quot;]
            The event to calculate the pitch of

        Returns
        -------
        float
            The average pitch of the event in Hz
        """

        # Interpolate pitch 0 values for avg pitch
        time = np.linspace(
            0, len(self.cache.pitch) / self.cache.pitch_sr, len(self.cache.pitch)
        )
        pitch_values_interp = np.copy(self.cache.pitch)
        pitch_values_interp[pitch_values_interp == 0] = np.nan
        pitch_values_interp = np.interp(
            time,
            time[
                ~np.isnan(pitch_values_interp)
            ],  # Don't include nan values in interpolation
            pitch_values_interp[
                ~np.isnan(pitch_values_interp)
            ],  # Don't include nan values in interpolation
        )

        # Convert signal samples to seconds and pitch samples
        start = event.start / self.asig.sr
        end = event.end / self.asig.sr

        # Interpolate the pitch
        pitch_values = np.interp(
            np.linspace(start, end, int((end - start) * self.cache.pitch_sr)),
            time,
            pitch_values_interp,
        )

        return np.mean(pitch_values)

    def plot_pitch(
        self,
        axes: plt.Axes = None,
        include_events: bool = True,
        xlabel: str = "Time (s)",
        **kwargs,
    ) -> None:
        """Plots the guessed pitch. This won't call plt.show(), allowing plot customization.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on, by default None.
            If None, a new figure will be created.
        include_events : bool, optional
            Whether or not to include the guessed events in the plot, by default True
        xlabel : str, optional
            The label of the x-axis, by default &quot;Time (s)&quot;
        **kwargs
            Additional arguments to be passed to matplotlib.pyplot.plot()
        """

        # Create a new axes if none is given
        if axes is None:
            axes = plt.subplot()

        # Plot the pitch
        time = np.linspace(
            0, len(self.cache.pitch) / self.cache.pitch_sr, len(self.cache.pitch)
        )
        plot_values = np.copy(self.cache.pitch)
        plot_values[plot_values == 0] = np.nan  # Replace 0 with nan to not plot it
        plot_values = librosa.hz_to_midi(plot_values)  # We want to plot midi values
        axes.plot(time, plot_values, **kwargs)

        # Label the axes
        axes.set_xlabel(xlabel)
        axes.set_ylabel("Pitch (MIDI)")

        # Add a grid for midi semitones in y-axis
        min_pitch = int(np.nanmin(plot_values)) - 1
        max_pitch = int(np.nanmax(plot_values)) + 1
        axes.set_yticks(np.arange(min_pitch, max_pitch + 1, 1))
        axes.grid(True, axis="y")

        # Plot the events with average pitch as line
        if include_events:
            for event in self.cache.events:
                # Convert signal samples to seconds and pitch samples
                start = event.start / self.asig.sr
                end = event.end / self.asig.sr

                avg_pitch = self._avg_pitch(event)
                avg_pitch = librosa.hz_to_midi(avg_pitch)
                axes.plot(
                    [
                        start,
                        end,
                    ],
                    [avg_pitch, avg_pitch],
                    color="red",
                )

            # Add legend
            axes.legend(["Detected pitch", "Average pitch of event"])

    def print_events(self) -> None:
        """Prints the guessed events."""

        for event in self.cache.events:
            start_seconds = event.start / self.asig.sr
            end_seconds = event.end / self.asig.sr
            event_index = self.cache.events.index(event)
            avg_pitch = self._avg_pitch(event)

            print(
                f"Event {event_index}: {start_seconds:.2f}s - {end_seconds:.2f}s "
                f"({end_seconds - start_seconds:.2f}s) - {avg_pitch:.2f}Hz"
            )

    def to_json(self) -> str:
        """Converts the esig to a json string.

        Returns
        -------
        str
            The json string.
        """

        # Convert the signal of asig to a binary string
        signal = self.asig.sig.tobytes()
        signal_base64 = binascii.b2a_base64(signal).decode("utf-8")
        sample_rate = self.asig.sr

        return json.dumps(
            {
                "signal_base64": signal_base64,
                "num_channels": self.asig.channels,
                "sample_rate": sample_rate,
                "algorithm": self.algorithm,
                "max_vibrato_extent": self.max_vibrato_extent,
                "max_vibrato_inaccuracy": self.max_vibrato_inaccuracy,
                "min_event_length": self.min_event_length,
                "edits": [edit.to_json() for edit in self.edits],
            }
        )

    def export(self, path: str, time_from: float | None, time_to: float | None) -> None:
        """Exports the esig to a wav file.

        Parameters
        ----------
        path : str
            The path to export the wav file to.
        time_from : float, optional
            The time to start exporting from, in seconds, by default None.
            If None, the export will start from the beginning.
        time_to : float, optional
            The time to stop exporting at, in seconds, by default None.
            If None, the export will stop at the end.
        """

        # Convert the signal of asig in cache to a binary string
        signal = self.cache.asig.sig
        sample_rate = self.cache.asig.sr

        # Convert time to samples
        if time_from is None:
            time_from = 0
        if time_to is None:
            time_to = len(signal) / sample_rate
        time_from = int(time_from * sample_rate)
        time_to = int(time_to * sample_rate)

        # Clip the signal
        signal = signal[time_from:time_to]

        # Create the wav file
        scipy.io.wavfile.write(
            path,
            sample_rate,
            signal,
        )


class Event:
    """A event is a range of samples with a guessed pitch."""

    def __init__(self, start: int, end: int) -> None:
        """Creates a event object, from start to end, with pitch as the guessed pitch.

        Parameters
        ----------
        start : int
            The starting point of this event (inclusive), in samples.
        end : int
            The ending point of this event (exclusive), in samples.
        """

        self.start = start
        self.end = end


class Cache:
    """We apply edits to a copy of the original signal, and store the results here."""

    def __init__(self, esig: Type["Esig"]) -> None:
        """Creates a cache object, which stores the results of edits.

        Parameters
        ----------
        esig : Type[&quot;Esig&quot;]
            The esig object to create the cache for.
        """

        self.esig = esig  # Store the esig object
        self.asig = Asig(
            np.copy(esig.asig.sig), esig.asig.sr
        )  # The current version of the audio signal
        (
            pitch,
            pitch_sr,
            frame_size,
            frame_jump,
        ) = self._recalculate(
            0, len(self.asig.sig)
        )  # Calculate the pitch
        self.pitch = pitch  # The current version of the pitch
        self.pitch_sr = pitch_sr  # The sample rate of the pitch
        self.frame_size = frame_size  # The current frame size
        self.frame_jump = frame_jump  # The current frame jump

        # Calculate events
        self.events = self._guess_events(self.pitch, self.pitch_sr)

    def apply(self, edit: Type["Edit"]) -> None:
        """Applies the given edit to the cache.
        This applies the given edit on top of all previous edits.

        Parameters
        ----------
        edit : Type[&quot;Edit&quot;]
            The edit to apply.
        """

        edit.apply(self)

    def reapply(self) -> None:
        """Applies all edits of the esig object to the cache.
        This applies all edits on top of the original asig and pitch.
        """

        # Copy the original asig from esig
        self.asig = Asig(np.copy(self.esig.asig.sig), self.esig.asig.sr)

        # Calculate pitch and events
        (
            self.pitch,
            self.pitch_sr,
            self.frame_size,
            self.frame_jump,
        ) = self._recalculate(0, len(self.asig.sig))
        self.events = self._guess_events(self.pitch, self.pitch_sr)

        # Apply all edits
        for edit in self.esig.edits:
            self.apply(edit)

    def update(self, start: int, end: int) -> None:
        """Recalculates the pitch of the cache for the given sample range (in signal samples)

        Parameters
        ----------
        start : int
            The starting point of the range to recalculate (inclusive), in samples.
        end : int
            The ending point of the range to recalculate (exclusive), in samples.
        """

        # Make sure the range is at least 500ms long
        min_samples = int(self.asig.sr * 0.5)
        if end - start < min_samples:
            end = min(
                start + min_samples, len(self.asig.sig)
            )  # Make sure we don't go out of bounds
            if end - start < min_samples:
                start = max(
                    end - min_samples, 0
                )  # If we still don't have enough samples, we go to the other side

        (
            edited_pitch,
            _,
            _,
            _,
        ) = self._recalculate(
            start, end
        )  # Recalculate the pitch

        # We need to convert the ranges to pitch samples
        start_pitch = int(start * (self.pitch_sr / self.asig.sr))
        end_pitch = int(end * (self.pitch_sr / self.asig.sr))

        # The length of the edited pitch might differ slightly from the original part,
        # therefore we interpolate the edited pitch to match the length of the original part.
        target_length = end_pitch - start_pitch
        edited_pitch = np.interp(
            np.linspace(0, len(edited_pitch), target_length),
            np.arange(len(edited_pitch)),
            edited_pitch,
        )

        # Replace the edited part of the pitch
        self.pitch = np.concatenate(
            (
                self.pitch[:start_pitch],
                edited_pitch,
                self.pitch[end_pitch:],
            )
        )

    def _recalculate(self, start: int, end: int) -> tuple[np.ndarray, float, int, int]:
        """Recalculate the pitch of the current signal in the cache for the given sample range.

        Parameters
        ----------
        start : int
            The starting point of the range to recalculate (inclusive), in samples.
        end : int
            The ending point of the range to recalculate (exclusive), in samples.

        Returns
        -------
        tuple[np.ndarray, float, int, int]
            The pitch, pitch sample rate, frame size, and frame jump.

        Raises
        ------
        ValueError
            If the algorithm is invalid.
        """

        # Guess the pitch of the audio signal
        if self.esig.algorithm == "yaapt":
            edited_part = self.asig.sig[start:end]

            pitch, frame_size, frame_jump = self._guess_pitch_yaapt(
                edited_part, self.asig.sr
            )
            length = (
                len(edited_part) / self.asig.sr
            )  # Length of the audio signal (in seconds)
            pitch_sr = len(pitch) / length  # Pitch sampling rate
        else:
            raise ValueError("Invalid algorithm")

        return pitch, pitch_sr, frame_size, frame_jump

    def _guess_pitch_yaapt(
        self, sig: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, int, int]:
        """Guesses the pitch of an audio signal.

        Parameters
        ----------
        sig : np.ndarray
            The audio signal.
        sample_rate : int
            The sample rate of the audio signal.

        Returns
        -------
        tuple[np.ndarray, int, int]
            The pitch, frame size, and frame jump of the pitch.
        """

        # Create a SignalObj
        mono = librosa.to_mono(sig.T)
        signal = basic_tools.SignalObj(mono, sample_rate)

        # Apply YAAPT
        pitch_guess = pYAAPT.yaapt(
            signal, frame_length=30, tda_frame_length=40, f0_min=60, f0_max=600
        )

        return pitch_guess.samp_values, pitch_guess.frame_size, pitch_guess.frame_jump

    def _guess_events(self, pitch: np.ndarray, pitch_sr: float) -> list:
        """Guesses the events from the pitch.

        Parameters
        ----------
        pitch : np.ndarray
            The pitch of the audio signal.
        pitch_sr : float
            The sample rate of the pitch.

        Returns
        -------
        list
            The guessed events.
            This list can be incomplete, e.g. parts of the audio signal have no event assigned.
        """

        # We first define a event as a range of samples,
        # where the pitch is not too far away from the mean pitch of the range.
        ranges = []
        start = 0  # Inclusive
        end = 0  # Exclusive
        for i, current_pitch in enumerate(pitch):
            # Extend event by one sample.
            end = i

            end_event = False

            # If the pitch is 0, end the current event.
            if current_pitch == 0:
                end_event = True
            else:
                # Get the pitches in the current event.
                pitches = pitch[start:end]
                new_pitches = np.append(pitches, current_pitch)
                average_vibrato_rate = 5  # Hz
                sigma = pitch_sr / (average_vibrato_rate * 2)
                new_pitches_gaussian = scipy.ndimage.gaussian_filter1d(
                    new_pitches, sigma
                )

                # Calculate what the average pitch would be
                # if we added the current sample to the event.
                new_avg = np.mean(new_pitches)
                new_avg_midi = librosa.hz_to_midi(new_avg)
                semitone_freq_delta = (
                    librosa.midi_to_hz(new_avg_midi + 1) - new_avg
                )  # Hz difference between avg and one semitone higher
                max_freq_deviation = semitone_freq_delta * (
                    self.esig.max_vibrato_extent / 100
                )  # Max deviation in Hz

                # If adding the current sample to the event would cause the pitch difference
                # between the average pitch and any pitch in the event to be above the max,
                # end the current event and start a new one.
                if any(
                    abs(pitch - new_avg) > max_freq_deviation for pitch in new_pitches
                ):
                    end_event = True
                # We end the event if the average pitch is too far away
                # from the gaussian-smoothed pitch.
                elif any(
                    abs(pitch_gaussian - new_avg)
                    > max_freq_deviation * self.esig.max_vibrato_inaccuracy
                    for pitch_gaussian in new_pitches_gaussian
                ):
                    end_event = True
                # If we have reached the end of the signal, end the current event
                elif i == len(pitch) - 1:
                    end_event = True

            if end_event:
                # If the event is long enough, add it to the list of events before ending it
                if end - start > self.esig.min_event_length * pitch_sr:
                    ranges.append((start, end))

                start = i

        # Create the events
        events = []
        for start, end in ranges:
            # Convert events to sample ranges
            start = int(start * (self.asig.sr / pitch_sr))
            end = int(end * (self.asig.sr / pitch_sr))

            events.append(Event(start, end))

        return events


class Edit(ABC):
    """A non-destructive edit to an esig object."""

    def __init__(self, start: int, end: int) -> None:
        """Creates a non-destructive edit object for the given sample range.

        Parameters
        ----------
        start : int
            The starting point of this edit (inclusive), in samples.
        end : int
            The ending point of this edit (exclusive), in samples.
        """

        self.start = start
        self.end = end

    @abstractmethod
    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given cache.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

    @abstractmethod
    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """


class PitchChange(Edit):
    """Changes the pitch of a sample range."""

    def __init__(
        self,
        start: int,
        end: int,
        change: float,
        algorithm: str,
    ) -> None:
        """Creates a non-destructive pitch change for the given sample range.

        Parameters
        ----------
        start : int
            The starting point of this edit (inclusive), in samples.
        end : int
            The ending point of this edit (exclusive), in samples.
        change : float
            The amount of semitones to change the pitch by. 0.0 is no change.
        algorithm : str
            The algorithm to change the pitch with.
        """

        super().__init__(start, end)
        self.change = change

        if algorithm not in ["tdpsola"]:
            raise ValueError("Invalid algorithm")

        self.algorithm = algorithm

    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given esig object.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

        if self.algorithm == "tdpsola":
            # The range is in signal samples, we need to convert it to pitch samples
            factor = len(cache.pitch) / len(cache.asig.sig)
            start = int(self.start * factor)
            end = int(self.end * factor)

            # Calculate the new pitch contour,
            # i.e. the pitch contour shifted by the shift factor for the given range
            changed_pitch = np.copy(cache.pitch)
            for i in range(start, end):
                # Get current pitch and adapt by the change factor
                current_pitch = changed_pitch[i]
                current_pitch = librosa.hz_to_midi(current_pitch)
                current_pitch += self.change
                current_pitch = librosa.midi_to_hz(current_pitch)
                changed_pitch[i] = current_pitch

            # Apply the pitch change
            cache.asig.sig = tsm.tdpsola(
                cache.asig.sig.T,
                cache.asig.sr,
                src_f0=cache.pitch,
                tgt_f0=changed_pitch,
                p_hop_size=cache.frame_jump,
                p_win_size=cache.frame_size,
            ).T  # Tdpsola returns (channels, samples) instead of (samples, channels)
        else:
            raise ValueError("Invalid algorithm")

        # Recalculate the pitch
        cache.update(self.start, self.end)

    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """

        return {
            "type": "pitch_change",
            "start": self.start,
            "end": self.end,
            "change": self.change,
            "algorithm": self.algorithm,
        }


class LengthChange(Edit):
    """Changes the length of a sample range."""

    def __init__(
        self,
        start: int,
        end: int,
        stretch_factor: float,
        algorithm: str,
    ) -> None:
        """Creates a non-destructive length change for the given sample range.

        Parameters
        ----------
        start : int
            The starting point of this edit (inclusive), in samples.
        end : int
            The ending point of this edit (exclusive), in samples.
        stretch_factor : float
            The factor to stretch the length by. 1.0 is no change.
        algorithm : str
            The algorithm to change the length with.
        """

        super().__init__(start, end)
        self.stretch_factor = stretch_factor

        if algorithm not in ["wsola"]:
            raise ValueError("Invalid algorithm")

        self.algorithm = algorithm

    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given esig object.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

        if self.algorithm == "wsola":
            # Save the parts before and after the edit
            before = cache.asig.sig[: self.start]
            after = cache.asig.sig[self.end :]

            # Apply the length change
            edit = tsm.wsola(
                cache.asig.sig[self.start : self.end].T,
                self.stretch_factor,
            ).T  # Wsola returns (channels, samples) instead of (samples, channels)

            # Reconstruct the signal
            cache.asig = Asig(np.concatenate((before, edit, after)), cache.asig.sr)

            # Update events to match the new length
            for event in cache.events:
                # If event start is after or during the edit, move it
                if event.start >= self.start:
                    # Find percentage of event in the edit
                    event_percentage = (event.start - self.start) / (
                        self.end - self.start
                    )

                    # Limit percentage to 1 (when event is at the end of the edit or after)
                    event_percentage = min(1, event_percentage)

                    # Find change in length of the edit
                    edit_length_change = len(edit) - (self.end - self.start)

                    # Move the event by the change in length
                    event.start += int(event_percentage * edit_length_change)

                # If event end is after or during the edit, move it
                if event.end >= self.start:
                    # Find percentage of event in the edit
                    event_percentage = (event.end - self.start) / (
                        self.end - self.start
                    )

                    # Limit percentage to 1 (when event is at the end of the edit or after)
                    event_percentage = min(1, event_percentage)

                    # Find change in length of the edit
                    edit_length_change = len(edit) - (self.end - self.start)

                    # Move the event by the change in length
                    event.end += int(event_percentage * edit_length_change)

        else:
            raise ValueError("Invalid algorithm")

        # Insert empty pitch samples for the edit
        start_pitch = int(self.start * (cache.pitch_sr / cache.asig.sr))
        end_pitch = int(self.end * (cache.pitch_sr / cache.asig.sr))
        pitch_before = cache.pitch[:start_pitch]
        pitch_after = cache.pitch[end_pitch:]
        edit_length_pitch = int(len(edit) * (cache.pitch_sr / cache.asig.sr))
        cache.pitch = np.concatenate(
            (pitch_before, [0] * edit_length_pitch, pitch_after)
        )

        # Recalculate the pitch
        edit_start = self.start
        edit_end = self.start + len(edit)
        cache.update(edit_start, edit_end)

    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """

        return {
            "type": "length_change",
            "start": self.start,
            "end": self.end,
            "stretch_factor": self.stretch_factor,
            "algorithm": self.algorithm,
        }


class EventModification(Edit):
    """An edit to manually modify an event.
    Possible modifications are:
    - Change the boundaries of the event (e.g. to make it longer)
    - Change the position of the event (e.g. to move it 20ms forward)
    """

    def __init__(
        self,
        start: int,
        end: int,
        new_start: int,
        new_end: int,
        offset: int,
    ) -> None:
        """Creates an event modification edit.
        The new boundaries of the event are applied first, then the offset is applied.
        The event to be modified is the one that matches the start and end of this edit exactly.

        Parameters
        ----------
        start : int
            The starting point of this edit (inclusive), in samples.
        end : int
            The ending point of this edit (exclusive), in samples.
        new_start : int
            The new starting point of the event (inclusive), in samples.
        new_end : int
            The new ending point of the event (exclusive), in samples.
        offset : int
            The offset to move the event by, in samples.
        """

        super().__init__(start, end)
        self.new_start = new_start
        self.new_end = new_end
        self.offset = offset

    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given esig object.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

        # Find the event to modify
        event = None
        for cache_event in cache.events:
            if cache_event.start == self.start and cache_event.end == self.end:
                event = cache_event
                break

        # When no event is found, raise an error
        if event is None:
            raise ValueError("No event found to modify")

        # Modify the event
        event.start = self.new_start
        event.end = self.new_end
        event.start += self.offset
        event.end += self.offset

        # Update the cache
        cache.update(self.start, self.end)

    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """

        return {
            "type": "event_modification",
            "start": self.start,
            "end": self.end,
            "new_start": self.new_start,
            "new_end": self.new_end,
            "offset": self.offset,
        }


class EventCreation(Edit):
    """An edit to manually create an event."""

    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given esig object.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

        # Create the event
        event = Event(self.start, self.end)

        # Add the event to the cache.
        # We don't need to update the cache because the event itself
        # makes no changes to the audio signal
        cache.events.append(event)

    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """

        return {
            "type": "event_creation",
            "start": self.start,
            "end": self.end,
        }


class EventDeletion(Edit):
    """Deletes all events that match the start and end of this edit exactly."""

    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given esig object.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

        # Find the events to delete
        events_to_delete = []
        for cache_event in cache.events:
            if cache_event.start == self.start and cache_event.end == self.end:
                events_to_delete.append(cache_event)

        # Delete the events.
        # We don't need to update the cache because the events themselves
        # make no changes to the audio signal.
        for event in events_to_delete:
            cache.events.remove(event)

    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """

        return {
            "type": "event_deletion",
            "start": self.start,
            "end": self.end,
        }


class PitchCurveEdit(Edit):
    """A Changes the pitch of a sample range to a target pitch curve."""

    def __init__(
        self,
        start: int,
        end: int,
        target_pitch: np.ndarray,
        algorithm: str,
    ) -> None:
        """Creates a non-destructive pitch change for the given sample range.

        Parameters
        ----------
        start : int
            The starting point of this edit (inclusive), in samples.
        end : int
            The ending point of this edit (exclusive), in samples.
        target_pitch : np.ndarray
            The target pitch curve, given as a numpy array of tuples (time, pitch).
            The time starts at 0 (start of the edit) and ends at 1 (end of the edit),
            given as a fraction of the length of the edit.
            The pitch is given in semitones as the difference to the original pitch.
        algorithm : str
            The algorithm to change the pitch with.
        """

        super().__init__(start, end)
        self.target_pitch = target_pitch

        if algorithm not in ["tdpsola"]:
            raise ValueError("Invalid algorithm")

        self.algorithm = algorithm

    def apply(self, cache: Type["Cache"]) -> None:
        """Applies the edit to the given esig object.

        Parameters
        ----------
        cache : Type[&quot;Cache&quot;]
            The cache to apply the edit to.
        """

        if self.algorithm == "tdpsola":
            # The range is in signal samples, we need to convert it to pitch samples
            factor = len(cache.pitch) / len(cache.asig.sig)
            start = int(self.start * factor)
            end = int(self.end * factor)

            # Calculate the new pitch contour,
            # i.e. the pitch contour shifted by the semitone difference for the given range
            changed_pitch = np.copy(cache.pitch)
            for i in range(start, end):
                # Get current pitch and adapt by the change factor
                current_time = (i - start) / (end - start)
                pitch_delta = np.interp(
                    current_time,
                    self.target_pitch[:, 0],
                    self.target_pitch[:, 1],
                )

                # To convert semitones to Hz, we need to know the current pitch,
                # because the frequency difference between two semitones depends on the pitch.
                pitch = changed_pitch[i]
                pitch = librosa.hz_to_midi(pitch)
                changed_pitch[i] = librosa.midi_to_hz(pitch + pitch_delta)

            # Apply the pitch change
            cache.asig.sig = tsm.tdpsola(
                cache.asig.sig.T,
                cache.asig.sr,
                src_f0=cache.pitch,
                tgt_f0=changed_pitch,
                p_hop_size=cache.frame_jump,
                p_win_size=cache.frame_size,
            ).T
        else:
            raise ValueError("Invalid algorithm")

        # Recalculate the pitch
        cache.update(self.start, self.end)

    def to_json(self) -> dict:
        """Converts the edit to a json dict.

        Returns
        -------
        dict
            The json dict.
        """

        return {
            "type": "pitch_curve_edit",
            "start": self.start,
            "end": self.end,
            "target_pitch": binascii.b2a_base64(self.target_pitch.astype(dtype=np.float32).tobytes()).decode(
                "utf-8"
            ),
            "algorithm": self.algorithm,
        }
