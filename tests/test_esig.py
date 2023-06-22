"""Testing of the esig module."""
from unittest import TestCase
import os
import numpy as np
from pya import Asig, Esig

TESTFILE_STACCATO__PATH = os.path.join(
    os.path.dirname(__file__), "samples", "staccato.wav"
)
TESTFILE2_LEGATO_PATH = os.path.join(os.path.dirname(__file__), "samples", "legato.wav")
TESTFILE3_VOICE_PATH = os.path.join(os.path.dirname(__file__), "samples", "voice.wav")
TESTFILE4_PIANO_PATH = os.path.join(os.path.dirname(__file__), "samples", "piano.wav")


class TestEsig(TestCase):
    """Testing of the esig class."""

    def setUp(self):
        """Loading test wav files as asig objects and creating esig objects."""

        self.asig_test_staccato = Asig(TESTFILE_STACCATO__PATH)
        self.asig_test_legato = Asig(TESTFILE2_LEGATO_PATH)
        self.asig_test_voice = Asig(TESTFILE3_VOICE_PATH)
        self.asig_test_piano = Asig(TESTFILE4_PIANO_PATH)

        self.esig_test_staccato = Esig(self.asig_test_staccato)
        self.esig_test_legato = Esig(self.asig_test_legato)
        self.esig_test_voice = Esig(self.asig_test_voice)
        self.esig_test_piano = Esig(self.asig_test_piano)

    def test_esig_yaapt(self):
        """Testing the yaapt pitch detection algorithm."""

        esig1 = Esig(self.asig_test_staccato, algorithm="yaapt")
        esig2 = Esig(self.asig_test_legato, algorithm="yaapt")
        esig3 = Esig(self.asig_test_voice, algorithm="yaapt")
        esig4 = Esig(self.asig_test_piano, algorithm="yaapt")

        # Some pitches should be detected
        self.assertTrue(len(esig1.cache.pitch) > 0)
        self.assertTrue(len(esig2.cache.pitch) > 0)
        self.assertTrue(len(esig3.cache.pitch) > 0)
        self.assertTrue(len(esig4.cache.pitch) > 0)

        # Some events should be detected
        self.assertTrue(len(esig1.cache.events) > 0)
        self.assertTrue(len(esig2.cache.events) > 0)
        self.assertTrue(len(esig3.cache.events) > 0)
        self.assertTrue(len(esig4.cache.events) > 0)

    def test_esig_plot_pitch(self):
        """Testing the plot_pitch method."""

        self.esig_test_staccato.plot_pitch()
        self.esig_test_legato.plot_pitch()
        self.esig_test_voice.plot_pitch()
        self.esig_test_piano.plot_pitch()

    def test_esig_change_length(self):
        """Testing length change."""

        events_staccato = self.esig_test_staccato.cache.events
        self.esig_test_staccato.change_length(0, 5, 1.5)
        self.assertTrue(
            len(self.esig_test_staccato.cache.events) == len(events_staccato)
        )

        events_legato = self.esig_test_legato.cache.events
        self.esig_test_legato.change_length(0, 5, 1.5)
        self.assertTrue(len(self.esig_test_legato.cache.events) == len(events_legato))

        events_voice = self.esig_test_voice.cache.events
        self.esig_test_voice.change_length(0, 5, 1.5)
        self.assertTrue(len(self.esig_test_voice.cache.events) == len(events_voice))

        events_piano = self.esig_test_piano.cache.events
        self.esig_test_piano.change_length(0, 5, 1.5)
        self.assertTrue(len(self.esig_test_piano.cache.events) == len(events_piano))

    def test_esig_change_pitch(self):
        """Tests changing pitch of an esig."""

        pitch_before_staccato = np.copy(self.esig_test_staccato.cache.pitch)
        pitch_before_legato = np.copy(self.esig_test_legato.cache.pitch)
        pitch_before_voice = np.copy(self.esig_test_voice.cache.pitch)
        pitch_before_piano = np.copy(self.esig_test_piano.cache.pitch)

        events_before_staccato = np.copy(self.esig_test_staccato.cache.events)
        events_before_legato = np.copy(self.esig_test_legato.cache.events)
        events_before_voice = np.copy(self.esig_test_voice.cache.events)
        events_before_piano = np.copy(self.esig_test_piano.cache.events)

        self.esig_test_staccato.change_pitch(0, 5, 1)
        self.esig_test_legato.change_pitch(0, 5, 1)
        self.esig_test_voice.change_pitch(0, 5, 1)
        self.esig_test_piano.change_pitch(0, 5, 1)

        # The pitches should change
        self.assertTrue(
            not np.array_equal(
                pitch_before_staccato, self.esig_test_staccato.cache.pitch
            )
        )
        self.assertTrue(
            not np.array_equal(pitch_before_legato, self.esig_test_legato.cache.pitch)
        )
        self.assertTrue(
            not np.array_equal(pitch_before_voice, self.esig_test_voice.cache.pitch)
        )
        self.assertTrue(
            not np.array_equal(pitch_before_piano, self.esig_test_piano.cache.pitch)
        )

        # The events should not change
        self.assertTrue(
            np.array_equal(events_before_staccato, self.esig_test_staccato.cache.events)
        )
        self.assertTrue(
            np.array_equal(events_before_legato, self.esig_test_legato.cache.events)
        )
        self.assertTrue(
            np.array_equal(events_before_voice, self.esig_test_voice.cache.events)
        )
        self.assertTrue(
            np.array_equal(events_before_piano, self.esig_test_piano.cache.events)
        )

    def test_esig_change_event_pitch(self):
        """Tests the change_event_pitch method, making sure that only the event pitch changes."""

        pitch_before_staccato = np.copy(self.esig_test_staccato.cache.pitch)
        pitch_before_legato = np.copy(self.esig_test_legato.cache.pitch)
        pitch_before_voice = np.copy(self.esig_test_voice.cache.pitch)
        pitch_before_piano = np.copy(self.esig_test_piano.cache.pitch)

        events_before_staccato = np.copy(self.esig_test_staccato.cache.events)
        events_before_legato = np.copy(self.esig_test_legato.cache.events)
        events_before_voice = np.copy(self.esig_test_voice.cache.events)
        events_before_piano = np.copy(self.esig_test_piano.cache.events)

        self.esig_test_staccato.change_event_pitch(0, 1)
        self.esig_test_legato.change_event_pitch(0, 1)
        self.esig_test_voice.change_event_pitch(0, 1)
        self.esig_test_piano.change_event_pitch(0, 1)

        # The pitches should change
        self.assertTrue(
            not np.array_equal(
                pitch_before_staccato, self.esig_test_staccato.cache.pitch
            )
        )
        self.assertTrue(
            not np.array_equal(pitch_before_legato, self.esig_test_legato.cache.pitch)
        )
        self.assertTrue(
            not np.array_equal(pitch_before_voice, self.esig_test_voice.cache.pitch)
        )
        self.assertTrue(
            not np.array_equal(pitch_before_piano, self.esig_test_piano.cache.pitch)
        )

        # The events should not change
        self.assertTrue(
            np.array_equal(events_before_staccato, self.esig_test_staccato.cache.events)
        )
        self.assertTrue(
            np.array_equal(events_before_legato, self.esig_test_legato.cache.events)
        )
        self.assertTrue(
            np.array_equal(events_before_voice, self.esig_test_voice.cache.events)
        )
        self.assertTrue(
            np.array_equal(events_before_piano, self.esig_test_piano.cache.events)
        )

    def test_esig_modify_event(self):
        """Tests the event modification function."""

        events_before_staccato = []
        events_before_legato = []
        events_before_voice = []
        events_before_piano = []

        # Copy the events
        for event in self.esig_test_staccato.cache.events:
            events_before_staccato.append(
                (
                    event.start,
                    event.end,
                )
            )
        for event in self.esig_test_legato.cache.events:
            events_before_legato.append(
                (
                    event.start,
                    event.end,
                )
            )
        for event in self.esig_test_voice.cache.events:
            events_before_voice.append(
                (
                    event.start,
                    event.end,
                )
            )
        for event in self.esig_test_piano.cache.events:
            events_before_piano.append(
                (
                    event.start,
                    event.end,
                )
            )

        self.esig_test_staccato.modify_event(0, None, None, 2)
        self.esig_test_legato.modify_event(0, None, None, 2)
        self.esig_test_voice.modify_event(0, None, None, 2)
        self.esig_test_piano.modify_event(0, None, None, 2)

        event_1_start_seconds = (
            self.esig_test_staccato.cache.events[0].start
            / self.esig_test_staccato.cache.asig.sr
        )
        event_1_end_seconds = (
            self.esig_test_staccato.cache.events[0].end
            / self.esig_test_staccato.cache.asig.sr
        )

        self.esig_test_staccato.modify_event(
            1, event_1_start_seconds + 0.1, event_1_end_seconds + 0.2, None
        )
        self.esig_test_legato.modify_event(
            1, event_1_start_seconds + 0.1, event_1_end_seconds + 0.2, None
        )
        self.esig_test_voice.modify_event(
            1, event_1_start_seconds + 0.1, event_1_end_seconds + 0.2, None
        )
        self.esig_test_piano.modify_event(
            1, event_1_start_seconds + 0.1, event_1_end_seconds + 0.2, None
        )

        # The events should change
        self.assertTrue(
            not np.array_equal(
                events_before_staccato,
                [
                    (event.start, event.end)
                    for event in self.esig_test_staccato.cache.events
                ],
            )
        )
        self.assertTrue(
            not np.array_equal(
                events_before_legato,
                [
                    (event.start, event.end)
                    for event in self.esig_test_legato.cache.events
                ],
            )
        )
        self.assertTrue(
            not np.array_equal(
                events_before_voice,
                [
                    (event.start, event.end)
                    for event in self.esig_test_voice.cache.events
                ],
            )
        )
        self.assertTrue(
            not np.array_equal(
                events_before_piano,
                [
                    (event.start, event.end)
                    for event in self.esig_test_piano.cache.events
                ],
            )
        )

    def test_esig_create_event(self):
        """Tests the event creation function
        by creating new events and looking at the cache event list.
        """

        staccato_event_count = len(self.esig_test_staccato.cache.events)
        legato_event_count = len(self.esig_test_legato.cache.events)
        voice_event_count = len(self.esig_test_voice.cache.events)
        piano_event_count = len(self.esig_test_piano.cache.events)

        self.esig_test_staccato.create_event(0.1, 1.1)
        self.esig_test_legato.create_event(0.1, 1.1)
        self.esig_test_voice.create_event(0.1, 1.1)
        self.esig_test_piano.create_event(0.1, 1.1)

        self.assertTrue(
            len(self.esig_test_staccato.cache.events) == staccato_event_count + 1
        )
        self.assertTrue(
            len(self.esig_test_legato.cache.events) == legato_event_count + 1
        )
        self.assertTrue(len(self.esig_test_voice.cache.events) == voice_event_count + 1)
        self.assertTrue(len(self.esig_test_piano.cache.events) == piano_event_count + 1)

    def test_esig_delete_event(self):
        """Tests the event deletion function
        by deleting events and looking at the cache event list.
        """

        staccato_event_count = len(self.esig_test_staccato.cache.events)
        legato_event_count = len(self.esig_test_legato.cache.events)
        voice_event_count = len(self.esig_test_voice.cache.events)
        piano_event_count = len(self.esig_test_piano.cache.events)

        self.esig_test_staccato.delete_event(0)
        self.esig_test_legato.delete_event(0)
        self.esig_test_voice.delete_event(0)
        self.esig_test_piano.delete_event(0)

        self.assertTrue(
            len(self.esig_test_staccato.cache.events) == staccato_event_count - 1
        )
        self.assertTrue(
            len(self.esig_test_legato.cache.events) == legato_event_count - 1
        )
        self.assertTrue(len(self.esig_test_voice.cache.events) == voice_event_count - 1)
        self.assertTrue(len(self.esig_test_piano.cache.events) == piano_event_count - 1)

    def test_esig_undo(self):
        """Tests the undo function."""

        self.esig_test_staccato.change_pitch(0, 5, 1.5)
        self.esig_test_legato.change_pitch(0, 5, 1.5)
        self.esig_test_voice.change_pitch(0, 5, 1.5)
        self.esig_test_piano.change_pitch(0, 5, 1.5)

        sig_staccato = np.copy(self.esig_test_staccato.cache.pitch)
        sig_legato = np.copy(self.esig_test_legato.cache.pitch)
        sig_voice = np.copy(self.esig_test_voice.cache.pitch)
        sig_piano = np.copy(self.esig_test_piano.cache.pitch)

        self.esig_test_staccato.undo_last()
        self.esig_test_legato.undo_last()
        self.esig_test_voice.undo_last()
        self.esig_test_piano.undo_last()

        self.esig_test_staccato.change_pitch(0, 5, 1.5)
        self.esig_test_legato.change_pitch(0, 5, 1.5)
        self.esig_test_voice.change_pitch(0, 5, 1.5)
        self.esig_test_piano.change_pitch(0, 5, 1.5)

        self.assertTrue(
            np.array_equal(sig_staccato, self.esig_test_staccato.cache.pitch)
        )
        self.assertTrue(np.array_equal(sig_legato, self.esig_test_legato.cache.pitch))
        self.assertTrue(np.array_equal(sig_voice, self.esig_test_voice.cache.pitch))
        self.assertTrue(np.array_equal(sig_piano, self.esig_test_piano.cache.pitch))

    def test_print_events(self):
        """Tests the print_events method."""

        self.esig_test_staccato.print_events()
        self.esig_test_legato.print_events()
        self.esig_test_voice.print_events()
        self.esig_test_piano.print_events()

    def test_serialization(self):
        """Tests the to_json method and from_json constructor."""

        json_str_staccato = self.esig_test_staccato.to_json()
        json_str_legato = self.esig_test_legato.to_json()
        json_str_voice = self.esig_test_voice.to_json()
        json_str_piano = self.esig_test_piano.to_json()

        esig_staccato = Esig(json_str_staccato)
        esig_legato = Esig(json_str_legato)
        esig_voice = Esig(json_str_voice)
        esig_piano = Esig(json_str_piano)

        self.assertTrue(
            np.array_equal(
                self.esig_test_staccato.cache.pitch, esig_staccato.cache.pitch
            )
        )
        self.assertTrue(
            np.array_equal(self.esig_test_legato.cache.pitch, esig_legato.cache.pitch)
        )
        self.assertTrue(
            np.array_equal(self.esig_test_voice.cache.pitch, esig_voice.cache.pitch)
        )
        self.assertTrue(
            np.array_equal(self.esig_test_piano.cache.pitch, esig_piano.cache.pitch)
        )
