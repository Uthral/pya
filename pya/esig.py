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
            events,
            frame_size,
            frame_jump,
        ) = self._recalculate()  # Calculate the pitch and events
        self.pitch = pitch  # The current version of the pitch
        self.pitch_sr = pitch_sr  # The sample rate of the pitch
        self.events = events  # The current version of the events
        self.frame_size = frame_size  # The current frame size
        self.frame_jump = frame_jump  # The current frame jump

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

        # Apply all edits
        for edit in self.esig.edits:
            self.apply(edit)

    def update(self) -> None:
        """Recalculates the pitch and events of the current signal in the cache."""

        (
            pitch,
            pitch_sr,
            events,
            frame_size,
            frame_jump,
        ) = self._recalculate()
        self.pitch = pitch
        self.pitch_sr = pitch_sr
        self.events = events
        self.frame_size = frame_size
        self.frame_jump = frame_jump

    def _recalculate(self) -> tuple[np.ndarray, float, list, int, int]:
        """Recalculate the pitch and events of the current signal in the cache.

        Returns
        -------
        tuple[np.ndarray, float, list, int, int]
            The pitch, pitch sample rate, events, frame size, and frame jump.

        Raises
        ------
        ValueError
            If the algorithm is invalid.
        """

        # Guess the pitch of the audio signal
        if self.esig.algorithm == "yaapt":
            pitch, frame_size, frame_jump = self._guess_pitch_yaapt(self.asig)
            length = (
                len(self.asig.sig) / self.asig.sr
            )  # Length of the audio signal (in seconds)
            pitch_sr = len(pitch) / length  # Pitch sampling rate
        else:
            raise ValueError("Invalid algorithm")

        # Guess the events from the pitch
        # TODO Use events detected from original pitch
        events = self._guess_events(pitch, pitch_sr)

        return pitch, pitch_sr, events, frame_size, frame_jump

    def _guess_pitch_yaapt(self, asig: Type["Asig"]) -> tuple[np.ndarray, int, int]:
        """Guesses the pitch of an audio signal.

        Parameters
        ----------
        asig : Type[&quot;Asig&quot;]
            The signal to guess the pitch of.

        Returns
        -------
        tuple[np.ndarray, int, int]
            The pitch, frame size, and frame jump of the pitch.
        """

        # Create a SignalObj
        signal = basic_tools.SignalObj(asig.sig, asig.sr)

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
        shift_factor: float,
        algorithm: str,
    ) -> None:
        """Creates a non-destructive pitch change for the given sample range.

        Parameters
        ----------
        start : int
            The starting point of this edit (inclusive), in samples.
        end : int
            The ending point of this edit (exclusive), in samples.
        shift_factor : float
            The factor to shift the pitch by. 1.0 is no change.
        algorithm : str
            The algorithm to change the pitch with.
        """

        super().__init__(start, end)
        self.shift_factor = shift_factor

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
            changed_pitch[start:end] *= self.shift_factor

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
        cache.update()


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
        else:
            raise ValueError("Invalid algorithm")

        # Recalculate the pitch and events
        cache.update()
