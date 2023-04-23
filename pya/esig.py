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
from pya.asig import Asig
import scipy.signal


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
        max_vibrato_rate: float = 10.0,
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
        max_vibrato_rate : float
            The maximum vibrato rate in Hz.
            Voice vibrato is usually between 5 and 8 Hz.
        """

        self.asig = asig
        self.algorithm = algorithm
        self.max_vibrato_extent = max_vibrato_extent
        self.max_vibrato_rate = max_vibrato_rate
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
        pitches = None
        start = 0
        end = 0
        for i, current_pitch in enumerate(self.pitch):
            # If the pitch is 0, end the current note if there is one
            if current_pitch == 0:
                if pitches is not None:
                    end = i
                    ranges.append((start, end))
                    pitches = None

                continue

            # If we have no pitches, start a new note
            if pitches is None:
                pitches = [current_pitch]
                start = i
            else:
                new_avg = np.mean(pitches + [current_pitch])
                new_avg_midi = librosa.hz_to_midi(new_avg)
                semitone_freq_delta = (
                    librosa.midi_to_hz(new_avg_midi + 1) - new_avg
                )  # Hz difference between avg and one semitone higher
                max_freq_deviation = semitone_freq_delta * (
                    self.max_vibrato_extent / 100
                )  # Max deviation in Hz

                end_note = False

                # If adding the current pitch to the note would make any pitch difference
                # to the mean above the max deviation, end the current note and start a new one
                if any(abs(pitch - new_avg) > max_freq_deviation for pitch in pitches):
                    end_note = True

                # If the vibrato rate of the current note is above the max, end the current note
                # and start a new one, e.g. consider it as a seperate note instead of a vibrato
                peaks = scipy.signal.find_peaks(self.pitch, prominence=0.1)[0]
                peaks_in_note = [
                    peak for peak in peaks if peak >= start and peak <= i
                ]  # Peaks in the current note
                if len(peaks_in_note) > 1:
                    vibrato_rate_period = np.mean(
                        np.diff(peaks_in_note)
                    )  # Avg period between peaks (in samples)
                    vibrato_rate = 1 / (
                        vibrato_rate_period / self.pitch_sr
                    )  # Frequency of peaks
                    if vibrato_rate > self.max_vibrato_rate:
                        end_note = True

                if end_note:
                    end = i
                    ranges.append((start, end))
                    pitches = [current_pitch]
                    start = i
                # If the pitch is close enough, add it to the current note
                else:
                    pitches.append(current_pitch)
                    end = i

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
                axes.plot(
                    [note.start / self.pitch_sr, note.end / self.pitch_sr],
                    [self._average_note_pitch(note), self._average_note_pitch(note)],
                    color="red",
                )


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
