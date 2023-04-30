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
        min_note_length: float = 0.1,
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
            The maximum difference between the average pitch of a note to each pitch in the note,
            in cents (100 cents = 1 semitone).
            Voice vibrato is usually below 100 cents.
        max_vibrato_inaccuracy : float
            A factor (between 0 and 1) that determines how accurate
            the pitch of a note has to be within the note, when the signal has vibrato.
            A value near 0 means that the pitch has to be very accurate,
            e.g. the vibrato has to be very even.
        min_note_length : float
            The minimum length of a note in seconds.
            Notes shorter than this will be filtered out.
        """

        self.asig = asig
        self.algorithm = algorithm
        self.max_vibrato_extent = max_vibrato_extent
        self.max_vibrato_inaccuracy = max_vibrato_inaccuracy
        self.min_note_length = min_note_length
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

        # Guess the notes from the pitch
        self.notes = self._guess_notes()

    def _guess_notes(self) -> list:
        """Guesses the notes from the pitch.

        Returns
        -------
        list
            The guessed notes.
            This list can be incomplete, e.g. parts of the audio signal have no note assigned.
        """

        # We first define a note as a range of samples,
        # where the pitch is not too far away from the mean pitch of the range.
        ranges = []
        start = 0  # Inclusive
        end = 0  # Exclusive
        for i, current_pitch in enumerate(self.pitch):
            # Extend note by one sample.
            end = i

            end_note = False

            # If the pitch is 0, end the current note.
            if current_pitch == 0:
                end_note = True
            else:
                # Get the pitches in the current note.
                pitches = self.pitch[start:end]
                new_pitches = np.append(pitches, current_pitch)
                average_vibrato_rate = 5  # Hz
                sigma = self.pitch_sr / (average_vibrato_rate * 2)
                new_pitches_gaussian = scipy.ndimage.gaussian_filter1d(
                    new_pitches, sigma
                )

                # Calculate what the average pitch would be
                # if we added the current sample to the note.
                new_avg = np.mean(new_pitches)
                new_avg_midi = librosa.hz_to_midi(new_avg)
                semitone_freq_delta = (
                    librosa.midi_to_hz(new_avg_midi + 1) - new_avg
                )  # Hz difference between avg and one semitone higher
                max_freq_deviation = semitone_freq_delta * (
                    self.max_vibrato_extent / 100
                )  # Max deviation in Hz

                # If adding the current sample to the note would cause the pitch difference
                # between the average pitch and any pitch in the note to be above the max,
                # end the current note and start a new one.
                if any(
                    abs(pitch - new_avg) > max_freq_deviation for pitch in new_pitches
                ):
                    end_note = True
                # We end the note if the average pitch is too far away
                # from the gaussian-smoothed pitch.
                elif any(
                    abs(pitch_gaussian - new_avg)
                    > max_freq_deviation * self.max_vibrato_inaccuracy
                    for pitch_gaussian in new_pitches_gaussian
                ):
                    end_note = True
                # If we have reached the end of the signal, end the current note
                elif i == len(self.pitch) - 1:
                    end_note = True

            if end_note:
                # If the note is long enough, add it to the list of notes before ending it
                if end - start > self.min_note_length * self.pitch_sr:
                    ranges.append((start, end))

                start = i

        # Create the notes
        notes = []
        for start, end in ranges:
            pitch = np.mean(self.pitch[start:end])
            notes.append(Note(start, end, pitch))

        return notes

    def _average_note_pitch(self, note: Type["Note"]) -> float:
        """Calculates the average pitch of a note.

        Parameters
        ----------
        note : Type[&quot;Note&quot;]
            The note to calculate the average pitch of.

        Returns
        -------
        float
            The average pitch in Hz.
        """

        return np.mean(self.pitch[note.start : note.end])

    def plot_pitch(
        self,
        axes: plt.Axes = None,
        include_notes: bool = True,
        xlabel: str = "Time (s)",
        **kwargs
    ):
        """Plots the guessed pitch. This won't call plt.show(), allowing plot customization.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on, by default None.
            If None, a new figure will be created.
        include_notes : bool, optional
            Whether or not to include the guessed notes in the plot, by default True
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

        # Plot the notes with average pitch as line
        if include_notes:
            for note in self.notes:
                avg_pitch = self._average_note_pitch(note)
                axes.plot(
                    [note.start / self.pitch_sr, (note.end - 1) / self.pitch_sr],
                    [avg_pitch, avg_pitch],
                    color="red",
                )

            # Add legend
            axes.legend(["Detected pitch", "Average pitch of note"])


class Note:
    """A note is a range of samples with a guessed pitch."""

    def __init__(self, start: int, end: int, pitch: float) -> None:
        """Creates a note object, from start to end, with pitch as the guessed pitch.

        Parameters
        ----------
        start : int
            The starting point of this note (inclusive), in samples.
        end : int
            The ending point of this note (exclusive), in samples.
        pitch : float
            The guessed pitch of this note, in Hz.
        """

        self.start = start
        self.end = end
        self.pitch = pitch
