"""This module is used to create editable audio signals.

This module is used to create editable audio signals, which can be manipulated in time and pitch.
The editing is non-destructive,
meaning that the original audio signal is not modified and changes can be undone.
"""


from typing import Type
from amfm_decompy import basic_tools, pYAAPT
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.ndimage
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

        # Guess the pitch of the audio signal
        if algorithm == "yaapt":
            # Create a SignalObj
            signal = basic_tools.SignalObj(self.asig.sig, self.asig.sr)

            # Apply YAAPT
            pitch_guess = pYAAPT.yaapt(
                signal, frame_length=30, tda_frame_length=40, f0_min=60, f0_max=600
            )
            self.pitch = pitch_guess.samp_values
            length = signal.size / signal.fs  # Length of the signal in seconds
            self.pitch_sr = len(self.pitch) / length  # Pitch sampling rate
        else:
            raise ValueError("Invalid algorithm")

        # Guess the events from the pitch
        self.events = self._guess_events()

    def _guess_events(self) -> list:
        """Guesses the events from the pitch.

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
        for i, current_pitch in enumerate(self.pitch):
            # Extend event by one sample.
            end = i

            end_event = False

            # If the pitch is 0, end the current event.
            if current_pitch == 0:
                end_event = True
            else:
                # Get the pitches in the current event.
                pitches = self.pitch[start:end]
                new_pitches = np.append(pitches, current_pitch)
                average_vibrato_rate = 5  # Hz
                sigma = self.pitch_sr / (average_vibrato_rate * 2)
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
                    self.max_vibrato_extent / 100
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
                    > max_freq_deviation * self.max_vibrato_inaccuracy
                    for pitch_gaussian in new_pitches_gaussian
                ):
                    end_event = True
                # If we have reached the end of the signal, end the current event
                elif i == len(self.pitch) - 1:
                    end_event = True

            if end_event:
                # If the event is long enough, add it to the list of events before ending it
                if end - start > self.min_event_length * self.pitch_sr:
                    ranges.append((start, end))

                start = i

        # Create the events
        events = []
        for start, end in ranges:
            events.append(Event(start, end))

        return events

    def _average_event_pitch(self, event: Type["Event"]) -> float:
        """Calculates the average pitch of a event.

        Parameters
        ----------
        event : Type[&quot;Event&quot;]
            The event to calculate the average pitch of.

        Returns
        -------
        float
            The average pitch in Hz.
        """

        return np.mean(self.pitch[event.start : event.end])

    def plot_pitch(
        self,
        axes: plt.Axes = None,
        include_events: bool = True,
        xlabel: str = "Time (s)",
        **kwargs
    ):
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
        time = np.linspace(0, len(self.pitch) / self.pitch_sr, len(self.pitch))
        axes.plot(time, self.pitch, **kwargs)

        # Label the axes
        axes.set_xlabel(xlabel)
        axes.set_ylabel("Pitch (Hz)")

        # Plot the events with average pitch as line
        if include_events:
            for event in self.events:
                avg_pitch = self._average_event_pitch(event)
                axes.plot(
                    [event.start / self.pitch_sr, (event.end - 1) / self.pitch_sr],
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
