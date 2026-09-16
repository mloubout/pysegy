"""
Shared data structures for the minimal Python implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Byte locations for binary file header fields
FH_BYTE2SAMPLE: Dict[str, int] = {
    "Job": 3200,
    "Line": 3204,
    "Reel": 3208,
    "DataTracePerEnsemble": 3212,
    "AuxiliaryTracePerEnsemble": 3214,
    "dt": 3216,
    "dtOrig": 3218,
    "ns": 3220,
    "nsOrig": 3222,
    "DataSampleFormat": 3224,
    "EnsembleFold": 3226,
    "TraceSorting": 3228,
    "VerticalSumCode": 3230,
    "SweepFrequencyStart": 3232,
    "SweepFrequencyEnd": 3234,
    "SweepLength": 3236,
    "SweepType": 3238,
    "SweepChannel": 3240,
    "SweepTaperlengthStart": 3242,
    "SweepTaperLengthEnd": 3244,
    "TaperType": 3246,
    "CorrelatedDataTraces": 3248,
    "BinaryGain": 3250,
    "AmplitudeRecoveryMethod": 3252,
    "MeasurementSystem": 3254,
    "ImpulseSignalPolarity": 3256,
    "VibratoryPolarityCode": 3258,
    # Unassigned by the standard: rev 2 defines the binary header through
    # byte 3296 and leaves 3297 onwards free.  A scalar for the sample
    # interval, which the format itself gives only a two-byte word: a depth
    # model coarser than 65.535 m has no other way to say its step.  Read
    # like the coordinate scalars -- positive multiplies, negative divides,
    # zero means the interval stands as written -- so a file that does not
    # set it, which is every file written before this, is unaffected.
    "SampleIntervalScalar": 3298,
    "SegyFormatRevisionNumber": 3500,
    "FixedLengthTraceFlag": 3502,
    "NumberOfExtTextualHeaders": 3504,
}

# Byte ranges for trace header fields. Each entry maps a header name to a
# ``(offset, size)`` tuple where ``size`` is either 2 or 4 bytes depending on
# the SEGY specification.
_TH_OFFSETS: Dict[str, int] = {
    "TraceNumWithinLine": 0,
    "TraceNumWithinFile": 4,
    "FieldRecord": 8,
    "TraceNumber": 12,
    "EnergySourcePoint": 16,
    "CDP": 20,
    "CDPTrace": 24,
    "TraceIDCode": 28,
    "NSummedTraces": 30,
    "NStackedTraces": 32,
    "DataUse": 34,
    "Offset": 36,
    "RecGroupElevation": 40,
    "SourceSurfaceElevation": 44,
    "SourceDepth": 48,
    "RecDatumElevation": 52,
    "SourceDatumElevation": 56,
    "SourceWaterDepth": 60,
    "GroupWaterDepth": 64,
    "ElevationScalar": 68,
    "RecSourceScalar": 70,
    "SourceX": 72,
    "SourceY": 76,
    "GroupX": 80,
    "GroupY": 84,
    "CoordUnits": 88,
    "WeatheringVelocity": 90,
    "SubWeatheringVelocity": 92,
    "UpholeTimeSource": 94,
    "UpholeTimeGroup": 96,
    "StaticCorrectionSource": 98,
    "StaticCorrectionGroup": 100,
    "TotalStaticApplied": 102,
    "LagTimeA": 104,
    "LagTimeB": 106,
    "DelayRecordingTime": 108,
    "MuteTimeStart": 110,
    "MuteTimeEnd": 112,
    "ns": 114,
    "dt": 116,
    "GainType": 118,
    "InstrumentGainConstant": 120,
    "InstrumntInitialGain": 122,
    "Correlated": 124,
    "SweepFrequencyStart": 126,
    "SweepFrequencyEnd": 128,
    "SweepLength": 130,
    "SweepType": 132,
    "SweepTraceTaperLengthStart": 134,
    "SweepTraceTaperLengthEnd": 136,
    "TaperType": 138,
    "AliasFilterFrequency": 140,
    "AliasFilterSlope": 142,
    "NotchFilterFrequency": 144,
    "NotchFilterSlope": 146,
    "LowCutFrequency": 148,
    "HighCutFrequency": 150,
    "LowCutSlope": 152,
    "HighCutSlope": 154,
    "Year": 156,
    "DayOfYear": 158,
    "HourOfDay": 160,
    "MinuteOfHour": 162,
    "SecondOfMinute": 164,
    "TimeCode": 166,
    "TraceWeightingFactor": 168,
    "GeophoneGroupNumberRoll": 170,
    "GeophoneGroupNumberTraceStart": 172,
    "GeophoneGroupNumberTraceEnd": 174,
    "GapSize": 176,
    "OverTravel": 178,
    "CDPX": 180,
    "CDPY": 184,
    "Inline3D": 188,
    "Crossline3D": 192,
    "ShotPoint": 196,
    "ShotPointScalar": 200,
    "TraceValueMeasurmentUnit": 202,
    "TransductionConstnatMantissa": 204,
    "TransductionConstantPower": 208,
    "TransductionUnit": 210,
    "TraceIdentifier": 212,
    "ScalarTraceHeader": 214,
    "SourceType": 216,
    "SourceEnergyDirectionMantissa": 218,
    "SourceEnergyDirectionExponent": 222,
    "SourceMeasurmentMantissa": 224,
    "SourceMeasurementExponent": 228,
    "SourceMeasurmentUnit": 230,
    "Unassigned1": 232,
    "Unassigned2": 236,
}

# Header fields stored as 4-byte integers
_TH_INT32_FIELDS = {
    "TraceNumWithinLine",
    "TraceNumWithinFile",
    "FieldRecord",
    "TraceNumber",
    "EnergySourcePoint",
    "CDP",
    "CDPTrace",
    "Offset",
    "RecGroupElevation",
    "SourceSurfaceElevation",
    "SourceDepth",
    "RecDatumElevation",
    "SourceDatumElevation",
    "SourceWaterDepth",
    "GroupWaterDepth",
    "SourceX",
    "SourceY",
    "GroupX",
    "GroupY",
    "CDPX",
    "CDPY",
    "Inline3D",
    "Crossline3D",
    "ShotPoint",
    "TransductionConstnatMantissa",
    "SourceEnergyDirectionMantissa",
    "SourceMeasurmentMantissa",
    "Unassigned1",
    "Unassigned2",
}

# Final mapping of header field to (offset, size)
TH_BYTE2SAMPLE: Dict[str, Tuple[int, int]] = {
    k: (off, 4 if k in _TH_INT32_FIELDS else 2)
    for k, off in _TH_OFFSETS.items()
}

# Discard private constants from namespace
del _TH_OFFSETS
del _TH_INT32_FIELDS

FH_FIELDS = list(FH_BYTE2SAMPLE.keys())
TH_FIELDS = list(TH_BYTE2SAMPLE.keys())


@dataclass
class BinaryFileHeader:
    """
    Container for parsed binary file header values.
    """

    values: Dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in FH_FIELDS}
    )
    keys_loaded: List[str] = field(default_factory=lambda: list(FH_FIELDS))

    def __getattr__(self, name):
        """
        Return the value for ``name`` from ``values``.
        """
        if name in self.values:
            return self.values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """
        Set ``name`` in ``values`` when it is a header field.
        """
        if name in {"values", "keys_loaded"} or name not in FH_FIELDS:
            super().__setattr__(name, value)
        else:
            self.values[name] = int(value)
            if name not in self.keys_loaded:
                self.keys_loaded.append(name)

    def __str__(self):
        """
        Return a multi-line representation of loaded fields.
        """
        lines = ["BinaryFileHeader:"]
        fields = self.keys_loaded or FH_FIELDS
        for k in fields:
            lines.append(f"    {k:30s}: {self.values[k]:9d}")
        return "\n".join(lines)

    __repr__ = __str__

    def __getstate__(self):
        return {"values": self.values, "keys_loaded": self.keys_loaded}

    def __setstate__(self, state):
        super().__setattr__("values", state["values"])
        super().__setattr__("keys_loaded", state["keys_loaded"])


@dataclass
class FileHeader:
    """
    Combined textual and binary file header.
    """

    th: bytes = b" " * 3200
    bfh: BinaryFileHeader = field(default_factory=BinaryFileHeader)

    def __str__(self) -> str:
        """
        Return a readable representation of the header.
        """
        return str(self.bfh)

    __repr__ = __str__


@dataclass
class BinaryTraceHeader:
    """
    Container for parsed binary trace header values.
    """

    values: Dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in TH_FIELDS}
    )
    keys_loaded: List[str] = field(default_factory=list)

    def __getattr__(self, name):
        """
        Return the header value ``name``.
        """
        if name in self.values:
            return self.values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """
        Assign ``value`` to ``name`` when valid.
        """
        if name in {"values", "keys_loaded"} or name not in TH_FIELDS:
            super().__setattr__(name, value)
        else:
            self.values[name] = int(value)
            if name not in self.keys_loaded:
                self.keys_loaded.append(name)

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return {"values": self.values, "keys_loaded": self.keys_loaded}

    def __setstate__(self, state):
        super().__setattr__("values", state["values"])
        super().__setattr__("keys_loaded", state["keys_loaded"])

    def __str__(self):
        """
        Return a multi-line representation of loaded fields.
        """
        lines = ["BinaryTraceHeader:"]
        fields = self.keys_loaded or TH_FIELDS
        for k in fields:
            lines.append(f"    {k:30s}: {self.values[k]:9d}")
        return "\n".join(lines)


@dataclass
class SeisBlock:
    """
    In-memory representation of a SEGY dataset.
    """

    fileheader: FileHeader
    traceheaders: List[BinaryTraceHeader]
    data: List[List[float]]

    def __len__(self) -> int:
        return len(self.traceheaders)

    def __str__(self) -> str:
        lines = ["SeisBlock:"]
        lines.append(f"    traces: {len(self.traceheaders)}")
        lines.append(f"    ns: {self.fileheader.bfh.ns}")
        lines.append(f"    dt: {self.fileheader.bfh.dt}")
        return "\n".join(lines)

    __repr__ = __str__
