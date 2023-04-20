"""This module is used to create editable audio signals.

This module is used to create editable audio signals, which can be manipulated in time and pitch.
The editing is non-destructive,
meaning that the original audio signal is not modified and changes can be undone.
"""


from typing import Type
from amfm_decompy import basic_tools, pYAAPT
import numpy as np
import matplotlib.pyplot as plt
from pya.asig import Asig


class Esig:
    """The main class for editable audio signals.

    This class represents an editable audio signal, which can be manipulated in time and pitch.
    To allow non-destructive editing, the unmodified audio signal is stored as an Asig object.
    """

    NOTE_MAX_PITCH_DELTA = 5

    def __init__(self, asig: Asig, algorithm: str = "yaapt") -> None:
        """Creates a new editable audio signal from an existing audio signal.

        Parameters
        ----------
        asig : Asig
            The signal to be edited.
        algorithm : str
            The algorithm to be used to guess the pitch of the audio signal.
            Possible values are: 'yaapt'
        """

        self.asig = asig
        self.algorithm = algorithm
        self.edits = []

        # Guess the pitch of the audio signal
        if algorithm == "yaapt":
            # Create a SignalObj
            signal = basic_tools.SignalObj(self.asig.sig, self.asig.sr)

            # Apply YAAPT
            pitch_guess = pYAAPT.yaapt(
                signal, frame_length=30, tda_frame_length=40, f0_min=75, f0_max=600
            )
            self.pitch = pitch_guess.samp_values
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
            # If we have no pitches yet but a current pitch, start a new note
            if pitches is None:
                if current_pitch > 0:
                    pitches = [current_pitch]
                    start = i
            else:
                new_avg = np.mean(pitches + [current_pitch])
                # If adding the current pitch to the note would make any pitch too far away,
                # end the current note and start a new one
                if any(
                    abs(pitch - new_avg) > self.NOTE_MAX_PITCH_DELTA
                    for pitch in pitches
                ):
                    end = i
                    ranges.append((start, end))
                    start = i
                    pitches = None
                # If the pitch is close enough, add it to the current note
                else:
                    pitches.append(current_pitch)

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

    def plot_pitch(self, include_notes: bool = True):
        """Plots the guessed pitch. This won't call plt.show(), allowing plot customization.

        Parameters
        ----------
        include_notes : bool, optional
            Whether or not to include the guessed notes in the plot, by default True
        """

        # Plot the pitch
        plt.plot(self.pitch)

        # Plot the notes with average pitch as line
        if include_notes:
            for note in self.notes:
                plt.plot(
                    [note.start, note.end],
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
            The starting point of this note, in samples.
        end : int
            The ending point of this note, in samples.
        pitch : float
            The guessed pitch of this note, in Hz.
        """

        self.start = start
        self.end = end
        self.pitch = pitch


if __name__ == "__main__":
    asig_test1 = Asig("./examples/samples/legato.wav")
    esig_test1 = Esig(asig_test1)

    asig_test2 = Asig("./examples/samples/bohemian_rhapsody_piano_monotone.wav")
    esig_test2 = Esig(asig_test2)

    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 2, 1)
    esig_test1.plot_pitch()
    ax2 = plt.subplot(1, 2, 2)
    esig_test2.plot_pitch()
    ax1.set_title("Legato")
    ax2.set_title("Bohemian Rhapsody")
    ax1.set_xlabel("Samples")
    ax2.set_xlabel("Samples")
    ax1.set_ylabel("Pitch (Hz)")
    ax2.set_ylabel("Pitch (Hz)")
    plt.show()
