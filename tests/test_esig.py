"""Testing of the esig module."""
from unittest import TestCase
import os
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

    def test_esig_events(self):
        """Testing events when applying edits."""

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
