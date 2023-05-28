"""This module is used to create editable audio signals.

This module is used to create editable audio signals, which can be manipulated in time and pitch.
The editing is non-destructive,
meaning that the original audio signal is not modified and changes can be undone.
"""


from typing import Type
from abc import ABC, abstractmethod
from amfm_decompy import basic_tools, pYAAPT
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.ndimage
import pytsmod as tsm
from pya.asig import Asig


class Esig:
    """The main class for editable audio signals.

    This class represents an editable audio signal, which can be manipulated in time and pitch.
    To allow non-destructive editing, the unmodified audio signal is stored as an Asig object.
    """

    def __init__(
        self,
        asig: Asig,
        algorithm: str = "yaapt",
        max_vibrato_extent: float = 40,
        max_vibrato_inaccuracy: float = 0.5,
        min_event_length: float = 0.1,
    ) -> None:
        """Creates a new editable audio signal from an existing audio signal.

        Parameters
        ----------
        asig : Asig
            The signal to be edited.
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

        self.asig = asig
        self.algorithm = algorithm
        self.max_vibrato_extent = max_vibrato_extent
        self.max_vibrato_inaccuracy = max_vibrato_inaccuracy
        self.min_event_length = min_event_length
        self.edits = []

        self.cache = Cache(self)  # Initialize the cache, storing the results of edits

    def undo_last(self) -> None:
        """Undos the last edit on this signal."""

        self.edits.pop()
        self.cache.reapply()  # We need to reapply all edits in the cache

    def change_pitch(
        self,
        start: float,
        end: float,
        shift_factor: float,
        algorithm: str = "tdpsola",
    ) -> None:
        """Changes the pitch of the given sample range by the given amount.

        Parameters
        ----------
        start : float
            The starting second to change (inclusive)
        end : float
            The ending second to change (exclusive)
        shift_factor : float
            The factor to change the pitch with.
            1.0 means no change.
        algorithm : str, optional
            The algorithm to change the pitch with, by default "tdpsola".
            Currently, only "tdpsola" is supported.
        """

        # Convert seconds to samples
        start = int(start * self.asig.sr)
        end = int(end * self.asig.sr)

        self.edits.append(PitchChange(start, end, shift_factor, algorithm))
        self.cache.apply(self.edits[-1])  # Apply the edit to the cache

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

    def plot_pitch(
        self,
        axes: plt.Axes = None,
        include_events: bool = True,
        xlabel: str = "Time (s)",
        **kwargs
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
        axes.plot(time, self.cache.pitch, **kwargs)

        # Label the axes
        axes.set_xlabel(xlabel)
        axes.set_ylabel("Pitch (Hz)")

        # Plot the events with average pitch as line
        if include_events:
            for event in self.cache.events:
                # Convert signal samples to pitch samples and seconds
                start = event.start / self.asig.sr
                end = event.end / self.asig.sr
                start_sample = int(start * self.cache.pitch_sr)
                end_sample = int(end * self.cache.pitch_sr)

                avg_pitch = np.mean(self.cache.pitch[start_sample:end_sample])
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

        # Recalculate pitch and events
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
        # therefore we need to interpolate the edited pitch to match the length of the original part.
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
        signal = basic_tools.SignalObj(sig, sample_rate)

        # Apply YAAPT
        pitch_guess = pYAAPT.yaapt(
            signal, frame_length=30, tda_frame_length=40, f0_min=60, f0_max=600
        )

        # We have to interpolate the pitch where the algorithm didn't guess it (i.e. 0)
        values = pitch_guess.samp_values
        for i, value in enumerate(values):
            if value == 0:
                # Find next non-zero value
                next_value = 0
                for j in range(i + 1, len(values)):
                    if values[j] != 0:
                        next_value = values[j]
                        break

                # Find last non-zero value
                last_value = 0
                for j in range(i - 1, -1, -1):
                    if values[j] != 0:
                        last_value = values[j]
                        break

                # Interpolate the value
                values[i] = round((next_value + last_value) / 2, 2)

        return values, pitch_guess.frame_size, pitch_guess.frame_jump

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

    @abstractmethod
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

        # Recalculate the pitch and events
        cache.update(self.start, self.end)


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
