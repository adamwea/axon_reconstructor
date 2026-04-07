from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.runner import run_spikesort_merge_stage, run_spikesort_stage


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_unit_helpers_prefer_spikeinterface_methods_for_counts_and_ids() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [101, 102, 102]

        def get_num_units(self):
            return 2

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()

    analyzer = _FakeAnalyzer()
    assert spikesort_runner._unit_ids_from_obj(analyzer) == ["101", "102"]
    assert spikesort_runner._unit_count(analyzer) == 2


def test_unit_helpers_handle_bool_ambiguous_unit_id_sequences() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _BoolAmbiguousSequence:
        def __init__(self, values):
            self._values = list(values)

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

        def __bool__(self):
            raise ValueError("ambiguous truth value")

    class _FakeSorting:
        def get_unit_ids(self):
            return _BoolAmbiguousSequence([201, 202])

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()

    analyzer = _FakeAnalyzer()
    assert spikesort_runner._unit_ids_from_obj(analyzer) == ["201", "202"]
    assert spikesort_runner._unit_count(analyzer) == 2


def test_extract_unit_locations_from_analyzer_computes_missing_extension() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeExtension:
        def get_data(self):
            return [[10.0, 20.0], [30.0, 40.0]]

    class _FakeSorting:
        def get_unit_ids(self):
            return [11, 12]

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()
            self._has_unit_locations = False
            self.compute_calls: list[object] = []

        def has_extension(self, name: str) -> bool:
            return bool(name == "unit_locations" and self._has_unit_locations)

        def compute(self, extension_name):
            self.compute_calls.append(extension_name)
            self._has_unit_locations = True

        def get_extension(self, name: str):
            assert name == "unit_locations"
            return _FakeExtension()

    analyzer = _FakeAnalyzer()
    locations, error = spikesort_runner._extract_unit_locations_from_analyzer(analyzer=analyzer)

    assert error is None
    assert analyzer.compute_calls == ["unit_locations"]
    assert locations == {
        "11": {"x_um": 10.0, "y_um": 20.0},
        "12": {"x_um": 30.0, "y_um": 40.0},
    }


def test_extract_unit_locations_from_analyzer_computes_dependency_chain() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeExtension:
        def get_data(self):
            return [[100.0, 200.0], [300.0, 400.0]]

    class _FakeSorting:
        def get_unit_ids(self):
            return [101, 202]

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()
            self._computed: set[str] = set()
            self.compute_calls: list[str] = []

        def has_extension(self, name: str) -> bool:
            return bool(name in self._computed)

        def compute(self, extension_name):
            if isinstance(extension_name, (list, tuple)):
                if len(extension_name) != 1:
                    raise AssertionError("expected single extension")
                extension_name = extension_name[0]

            name = str(extension_name)
            self.compute_calls.append(name)
            if name == "unit_locations" and not {"random_spikes", "waveforms", "templates"}.issubset(self._computed):
                raise AssertionError("Extension unit_locations requires templates to be computed first")
            if name == "templates" and not {"random_spikes", "waveforms"}.issubset(self._computed):
                raise AssertionError("Extension templates requires random_spikes|waveforms to be computed first")
            if name == "waveforms" and "random_spikes" not in self._computed:
                raise AssertionError("Extension waveforms requires random_spikes")
            self._computed.add(name)

        def get_extension(self, name: str):
            assert name == "unit_locations"
            if "unit_locations" not in self._computed:
                return None
            return _FakeExtension()

    analyzer = _FakeAnalyzer()
    locations, error = spikesort_runner._extract_unit_locations_from_analyzer(analyzer=analyzer)

    assert error is None
    assert locations == {
        "101": {"x_um": 100.0, "y_um": 200.0},
        "202": {"x_um": 300.0, "y_um": 400.0},
    }
    assert {"random_spikes", "waveforms", "templates", "unit_locations"}.issubset(set(analyzer.compute_calls))


def test_write_merge_unit_location_reports_inverts_y_axis(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.inverted = False

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            self.inverted = True

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 11.0, "y_um": 21.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=True,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_before_relpath="unit_locations_before_merge.png",
        merge_reports_2panel_after_write_png=True,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_after_relpath="unit_locations_after_merge.png",
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("before_unit_locations_count") == 2
    assert payload.get("after_unit_locations_count") == 1
    assert all(ax.inverted for ax in fake_plt.axes_created)


def test_write_merge_unit_location_reports_labels_unit_ids_when_enabled(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.text_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            self.text_calls.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(text),
                    "ha": ha,
                    "va": va,
                    "transform": transform,
                }
            )
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 11.0, "y_um": 21.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_label_pre_and_post_units=True,
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 2
    before_texts = [str(call.get("text", "")) for call in fake_plt.axes_created[0].text_calls]
    after_texts = [str(call.get("text", "")) for call in fake_plt.axes_created[1].text_calls]
    assert "1" in before_texts
    assert "2" in before_texts
    assert "10" in after_texts


def test_write_merge_unit_location_reports_plots_highlights_after_other_units_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
                "3": {"x_um": 50.0, "y_um": 60.0},
            }
        }
    }
    after_snapshot = {"analyzer": {"unit_locations_by_unit": {}}}
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=True,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_before_relpath="unit_locations_before_merge.png",
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_before_point_color="#7a7a7a",
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_before_color="#ff7f0e",
        merge_reports_2panel_highlight_plot_after_other_units=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["2", "3"], "post_unit_id": None}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 1
    scatter_calls = fake_plt.axes_created[0].scatter_calls
    assert len(scatter_calls) == 2
    assert scatter_calls[0].get("xs") == [10.0]
    assert scatter_calls[0].get("colors") == ["#7a7a7a"]
    assert scatter_calls[1].get("xs") == [30.0, 50.0]
    assert scatter_calls[1].get("colors") == ["#ff7f0e", "#ff7f0e"]


def test_write_merge_unit_location_reports_labels_only_affected_units_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.text_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            self.text_calls.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(text),
                    "ha": ha,
                    "va": va,
                    "transform": transform,
                }
            )
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "1": {"x_um": 10.0, "y_um": 20.0},
                "2": {"x_um": 30.0, "y_um": 40.0},
                "3": {"x_um": 50.0, "y_um": 60.0},
            }
        }
    }
    after_snapshot = {"analyzer": {"unit_locations_by_unit": {}}}
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_label_pre_and_post_units=False,
        merge_reports_2panel_before_write_png=True,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_before_relpath="unit_locations_before_merge.png",
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_before_color="#ff7f0e",
        merge_reports_2panel_highlight_label_affected_units=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["2", "3"], "post_unit_id": None}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert len(fake_plt.axes_created) == 1
    before_texts = [str(call.get("text", "")) for call in fake_plt.axes_created[0].text_calls]
    assert "1" not in before_texts
    assert "2" in before_texts
    assert "3" in before_texts


def test_write_merge_unit_location_reports_infers_after_highlight_when_post_unit_missing(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def get_cmap(self, name: str):
            def _cmap(_value: float):
                return (0.1, 0.2, 0.9, 1.0)

            return _cmap

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "190": {"x_um": 2653.33, "y_um": 2079.95},
                "195": {"x_um": 2653.45, "y_um": 2080.30},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "205": {"x_um": 2653.69, "y_um": 2079.56},
                "120": {"x_um": 100.0, "y_um": 100.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=True,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_after_relpath="unit_locations_after_merge.png",
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=True,
        merge_reports_2panel_highlight_palette="tab20",
        merge_reports_2panel_after_point_color="#7a7a7a",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["190", "195"], "post_unit_id": None}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("after_highlighted_units_count") == 1
    assert payload.get("after_highlighted_inferred_units_count") == 1
    assert len(fake_plt.axes_created) == 1
    scatter_colors = list(fake_plt.axes_created[0].scatter_calls[0].get("colors", []))
    assert any(color != "#7a7a7a" for color in scatter_colors)


def test_write_merge_unit_location_reports_does_not_highlight_premerge_ids_on_after_panel(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[dict[str, object]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            return None

        def set_ylim(self, limits):
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            self.scatter_calls.append(
                {
                    "xs": list(xs),
                    "ys": list(ys),
                    "colors": list(c),
                }
            )
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 10.0, "y_um": 10.0},
                "20": {"x_um": 20.0, "y_um": 20.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "20": {"x_um": 20.0, "y_um": 20.0},
                "31": {"x_um": 31.0, "y_um": 31.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=True,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_after_relpath="unit_locations_after_merge.png",
        merge_reports_2panel_write_png=False,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_after_color="#2ca02c",
        merge_reports_2panel_after_point_color="#7a7a7a",
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["10", "20"], "post_unit_id": "20"}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("after_highlighted_units_count") == 1
    assert len(fake_plt.axes_created) == 1
    scatter_colors = list(fake_plt.axes_created[0].scatter_calls[0].get("colors", []))
    assert scatter_colors == ["#7a7a7a", "#2ca02c"]


def test_write_merge_unit_location_reports_zoom_to_affected_units_uses_affected_extent(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace as _SimpleNamespace

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAxis:
        def __init__(self) -> None:
            self.x_limits: list[tuple[float, float]] = []
            self.y_limits: list[tuple[float, float]] = []

        def set_title(self, title):
            return None

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xlim(self, limits):
            self.x_limits.append((float(limits[0]), float(limits[1])))
            return None

        def set_ylim(self, limits):
            self.y_limits.append((float(limits[0]), float(limits[1])))
            return None

        def invert_yaxis(self):
            return None

        def set_aspect(self, aspect, adjustable=None):
            return None

        def grid(self, enabled, alpha=None):
            return None

        def scatter(self, xs, ys, s=None, alpha=None, c=None):
            return None

        def text(self, x, y, text, ha=None, va=None, transform=None):
            return None

        @property
        def transAxes(self):
            return object()

    class _FakeFigure:
        def savefig(self, path, dpi=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake", encoding="utf-8")

    class _FakePyplot:
        def __init__(self) -> None:
            self.axes_created: list[_FakeAxis] = []

        def subplots(self, nrows=1, ncols=1, figsize=None, constrained_layout=False):
            fig = _FakeFigure()
            if int(nrows) == 1 and int(ncols) == 1:
                ax = _FakeAxis()
                self.axes_created.append(ax)
                return fig, ax
            axes = [_FakeAxis() for _ in range(int(nrows) * int(ncols))]
            self.axes_created.extend(axes)
            return fig, axes

        def close(self, fig):
            return None

    fake_plt = _FakePyplot()
    fake_matplotlib = _SimpleNamespace(pyplot=fake_plt)

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    before_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "10": {"x_um": 10.0, "y_um": 10.0},
                "20": {"x_um": 20.0, "y_um": 20.0},
                "99": {"x_um": 1000.0, "y_um": 1000.0},
            }
        }
    }
    after_snapshot = {
        "analyzer": {
            "unit_locations_by_unit": {
                "101": {"x_um": 110.0, "y_um": 130.0},
                "150": {"x_um": 1500.0, "y_um": 1500.0},
            }
        }
    }
    stage_cfg = SimpleNamespace(
        merge_reports_2panel_before_write_png=False,
        merge_reports_2panel_before_write_svg=False,
        merge_reports_2panel_after_write_png=False,
        merge_reports_2panel_after_write_svg=False,
        merge_reports_2panel_write_png=True,
        merge_reports_2panel_write_svg=False,
        merge_reports_2panel_relpath="unit_locations_before_after_merge.png",
        merge_reports_2panel_highlight_merges_enabled=True,
        merge_reports_2panel_highlight_merges_linked=False,
        merge_reports_2panel_highlight_after_color="#2ca02c",
        merge_reports_2panel_zoom_to_affected_units=True,
    )

    payload = spikesort_runner._write_merge_unit_location_reports(
        merge_out_dir=tmp_path,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        applied_unit_mappings=[{"pre_unit_ids": ["10", "20"], "post_unit_id": "101"}],
        stage_config=stage_cfg,
    )

    assert payload.get("status") == "ok"
    assert payload.get("zoom_to_affected_units") is True
    assert payload.get("zoom_to_affected_units_applied") is True
    assert len(fake_plt.axes_created) == 2
    for ax in fake_plt.axes_created:
        assert ax.x_limits
        assert ax.y_limits
        x_min, x_max = ax.x_limits[0]
        y_min, y_max = ax.y_limits[0]
        assert x_min > 0.0
        assert x_max < 500.0
        assert y_min > 0.0
        assert y_max < 500.0


def test_build_applied_unit_mappings_uses_post_unit_hint_even_if_missing_from_post_snapshot() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    mappings = spikesort_runner._build_applied_unit_mappings(
        applied_operations=[
            {
                "method": "slay",
                "group_id": "202",
                "pre_unit_ids": ["190", "195"],
                "post_unit_id_hint": "202",
            }
        ],
        pre_analyzer_payload={
            "unit_ids": ["190", "195"],
            "unit_locations_by_unit": {
                "190": {"x_um": 2653.33, "y_um": 2079.95},
                "195": {"x_um": 2653.45, "y_um": 2080.30},
            },
        },
        post_analyzer_payload={
            "unit_ids": ["105", "120"],
            "unit_locations_by_unit": {
                "105": {"x_um": 2653.69, "y_um": 2079.56},
                "120": {"x_um": 100.0, "y_um": 100.0},
            },
        },
    )

    assert len(mappings) == 1
    assert mappings[0].get("post_unit_id") == "202"
    assert mappings[0].get("resolution") == "post_unit_hint_missing_in_post_snapshot"
    assert mappings[0].get("post_unit_id_in_post_snapshot") is False


def test_build_merge_metadata_summary_uses_before_after_set_delta_for_added_ids() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_auto_accept_merges=True,
        auto_merge_enabled=False,
        auto_merge_auto_accept_merges=False,
    )

    summary = spikesort_runner._build_merge_metadata_summary(
        requested_sequence_raw=["SLAy"],
        stage_config=stage_cfg,
        pre_snapshot={
            "sorter": {"available": True, "unit_ids": ["0", "22", "105"]},
            "analyzer": {"available": True, "unit_ids": ["0", "22", "105"]},
        },
        post_snapshot={
            "sorter": {"available": True, "unit_ids": ["105"]},
            "analyzer": {"available": True, "unit_ids": ["105"]},
        },
        applied_operations=[
            {
                "method": "slay",
                "group_id": "202",
                "pre_unit_ids": ["0", "22"],
                "post_unit_id_hint": "105",
            }
        ],
    )

    analyzer_delta = summary.get("delta", {}).get("analyzer", {})
    sorter_delta = summary.get("delta", {}).get("sorter", {})

    assert analyzer_delta.get("added_unit_ids_set_delta") == []
    assert analyzer_delta.get("added_unit_ids") == []
    assert analyzer_delta.get("added_unit_ids_source") == "set_delta"
    assert set(analyzer_delta.get("removed_unit_ids", [])) == {"0", "22"}
    assert analyzer_delta.get("merge_target_unit_ids_from_mappings") == ["105"]
    assert analyzer_delta.get("merge_tracking_validation", {}).get("added_ids_cover_mapping_targets") is False
    assert analyzer_delta.get("merge_tracking_validation", {}).get("targets_present_in_post_snapshot") is True

    assert sorter_delta.get("added_unit_ids_set_delta") == []
    assert sorter_delta.get("added_unit_ids") == []
    assert sorter_delta.get("added_unit_ids_source") == "set_delta"


def test_build_merge_metadata_summary_keeps_mapping_target_gaps_as_diagnostics_only() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_auto_accept_merges=True,
        auto_merge_enabled=False,
        auto_merge_auto_accept_merges=False,
    )

    summary = spikesort_runner._build_merge_metadata_summary(
        requested_sequence_raw=["SLAy"],
        stage_config=stage_cfg,
        pre_snapshot={
            "sorter": {"available": True, "unit_ids": ["0", "22", "190", "195"]},
            "analyzer": {"available": True, "unit_ids": ["0", "22", "190", "195"]},
        },
        post_snapshot={
            "sorter": {"available": True, "unit_ids": ["1", "2"]},
            "analyzer": {"available": True, "unit_ids": ["1", "2"]},
        },
        applied_operations=[
            {
                "method": "slay",
                "group_id": "202",
                "pre_unit_ids": ["190", "195"],
                "post_unit_id_hint": "202",
            },
            {
                "method": "slay",
                "group_id": "203",
                "pre_unit_ids": ["0", "22"],
                "post_unit_id_hint": "203",
            },
        ],
    )

    change_validation = summary.get("change_validation", {})
    analyzer_delta = summary.get("delta", {}).get("analyzer", {})

    assert change_validation.get("passes") is True
    assert change_validation.get("reason") == "ok"
    assert change_validation.get("merge_target_unit_ids") == ["202", "203"]
    assert set(change_validation.get("merge_target_ids_missing_from_analyzer_post", [])) == {"202", "203"}
    assert analyzer_delta.get("added_unit_ids") == ["1", "2"]


def test_compute_snapshot_unit_delta_uses_snapshot_unit_count_when_ids_missing() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    delta = spikesort_runner._compute_snapshot_unit_delta(
        before_payload={"available": True, "unit_count": 7, "unit_ids": []},
        after_payload={"available": True, "unit_count": 5, "unit_ids": []},
    )

    assert delta.get("compared") is True
    assert delta.get("changed") is True
    assert delta.get("before_unit_count") == 7
    assert delta.get("after_unit_count") == 5
    assert delta.get("before_unit_ids") == []
    assert delta.get("after_unit_ids") == []


def test_build_unit_diff_map_and_flat_map_supports_chained_merges() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    unit_diff_payload = {
        "before": {
            "analyzer": {
                "unit_ids": ["10", "11", "12", "13"],
                "unit_locations_by_unit": {
                    "10": {"x_um": 10.0, "y_um": 10.0},
                    "11": {"x_um": 11.0, "y_um": 11.0},
                    "12": {"x_um": 12.0, "y_um": 12.0},
                    "13": {"x_um": 13.0, "y_um": 13.0},
                },
            }
        },
        "after": {
            "analyzer": {
                "unit_ids": ["25"],
                "unit_locations_by_unit": {
                    "25": {"x_um": 25.0, "y_um": 25.0},
                },
            }
        },
        "applied_merge_operations": [
            {
                "method": "slay",
                "group_id": "g1",
                "pre_unit_ids": ["10", "11"],
            },
            {
                "method": "auto_merge",
                "group_id": "g2",
                "pre_unit_ids": ["12", "20"],
                "iteration": 1,
            },
            {
                "method": "auto_merge",
                "group_id": "g3",
                "pre_unit_ids": ["21", "13"],
                "iteration": 2,
            },
        ],
        "applied_unit_mappings": [
            {
                "method": "slay",
                "group_id": "g1",
                "pre_unit_ids": ["10", "11"],
                "post_unit_id": "20",
                "resolution": "post_unit_hint",
            },
            {
                "method": "auto_merge",
                "group_id": "g2",
                "pre_unit_ids": ["12", "20"],
                "iteration": 1,
                "post_unit_id": "21",
                "resolution": "post_unit_hint",
            },
            {
                "method": "auto_merge",
                "group_id": "g3",
                "pre_unit_ids": ["21", "13"],
                "iteration": 2,
                "post_unit_id": "25",
                "resolution": "post_unit_hint",
            },
        ],
    }

    op_map = spikesort_runner._build_unit_diff_map_payload(unit_diff_payload=unit_diff_payload)
    flat_map = spikesort_runner._build_unit_diff_map_flat_payload(
        unit_diff_map_payload=op_map,
        unit_diff_payload=unit_diff_payload,
    )

    assert op_map.get("summary", {}).get("n_operations") == 3
    assert flat_map.get("summary", {}).get("n_flat_groups") == 1
    groups = flat_map.get("groups", [])
    assert len(groups) == 1
    assert groups[0].get("final_post_unit_id") == "25"
    assert groups[0].get("primary_pre_unit_ids") == ["10", "11", "12", "13"]


def test_extract_plot_inputs_from_unit_diff_report_prefers_flattened_groups() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    unit_diff_payload = {
        "before": {"analyzer": {"unit_locations_by_unit": {"10": {"x_um": 10.0, "y_um": 10.0}}}},
        "after": {"analyzer": {"unit_locations_by_unit": {"25": {"x_um": 25.0, "y_um": 25.0}}}},
        "applied_unit_mappings": [
            {"method": "slay", "group_id": "g1", "pre_unit_ids": ["10"], "post_unit_id": "20"}
        ],
        "unit_diff_map_flat": {
            "groups": [
                {
                    "group_id": "flat_g1",
                    "final_post_unit_id": "25",
                    "primary_pre_unit_ids": ["10"],
                }
            ]
        },
    }

    before_snapshot, after_snapshot, mappings = spikesort_runner._extract_plot_inputs_from_unit_diff_report(
        unit_diff_payload=unit_diff_payload
    )

    assert isinstance(before_snapshot, dict)
    assert isinstance(after_snapshot, dict)
    assert len(mappings) == 1
    assert mappings[0].get("post_unit_id") == "25"
    assert mappings[0].get("pre_unit_ids") == ["10"]


def test_build_post_merge_unit_locations_payload_includes_ids_and_locations() -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    payload = spikesort_runner._build_post_merge_unit_locations_payload(
        post_snapshot={
            "analyzer": {
                "unit_ids": ["21", "22"],
                "unit_locations_by_unit": {
                    "21": {"x_um": 21.0, "y_um": 22.0},
                    "22": {"x_um": 22.0, "y_um": 23.0},
                },
            }
        }
    )

    assert payload.get("summary", {}).get("n_unit_ids") == 2
    assert payload.get("summary", {}).get("n_locations") == 2
    assert payload.get("unit_ids") == ["21", "22"]
    assert payload.get("unit_locations_by_unit", {}).get("21", {}).get("x_um") == 21.0


def test_capture_merge_state_snapshot_uses_analyzer_sorting_if_sorter_load_fails(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeSorting:
        def get_unit_ids(self):
            return [11, 12, 13]

        def get_num_units(self):
            return 3

    class _FakeAnalyzer:
        def __init__(self) -> None:
            self.sorting = _FakeSorting()

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    stage_output_root_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(
        spikesort_runner,
        "_load_sorting_from_sorter_output_dir",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cannot load sorter")),
    )
    monkeypatch.setattr(
        spikesort_runner,
        "_load_or_recompute_spikesort_analyzer",
        lambda **kwargs: (_FakeAnalyzer(), stage_output_root_dir / "analyzer_output", False),
    )

    snapshot = spikesort_runner._capture_merge_state_snapshot(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(sorter="kilosort4", slay_sorter_output_relpath=None),
        capture_label="after_merge",
        include_unit_locations=False,
        allow_analyzer_recompute=True,
    )

    assert snapshot.get("sorter", {}).get("available") is True
    assert snapshot.get("sorter", {}).get("load_error") is None
    assert snapshot.get("sorter", {}).get("unit_count") == 3
    assert snapshot.get("sorter", {}).get("unit_ids") == ["11", "12", "13"]


def test_resolve_sorter_output_dir_prefers_wrapper_with_spikeinterface_markers(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    wrapper_dir = well_out_dir / "spikesort_outputs" / "sorter_output"
    nested_dir = wrapper_dir / "sorter_output"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "params.py").write_text("n_channels_dat=1\n", encoding="utf-8")
    (wrapper_dir / "spikeinterface_params.json").write_text("{}", encoding="utf-8")

    resolved = spikesort_runner._resolve_sorter_output_dir(
        well_out_dir=well_out_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(slay_sorter_output_relpath=None),
    )

    assert resolved == wrapper_dir.resolve()


def test_load_sorting_from_sorter_output_dir_tries_wrapper_parent(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    wrapper_dir = tmp_path / "sorter_output"
    nested_dir = wrapper_dir / "sorter_output"
    nested_dir.mkdir(parents=True, exist_ok=True)

    class _FakeSI:
        def read_sorter_folder(self, folder, sorter_name=None):
            if Path(folder).resolve() == wrapper_dir.resolve():
                return "loaded_from_wrapper"
            raise RuntimeError("wrong folder")

        def load_extractor(self, folder):
            raise RuntimeError("not used")

    loaded = spikesort_runner._load_sorting_from_sorter_output_dir(
        si_module=_FakeSI(),
        sorter_output_dir=nested_dir,
        sorter_name="kilosort4",
    )

    assert loaded == "loaded_from_wrapper"


def test_load_sorting_from_sorter_output_dir_prefers_kilosort_raw_ids(tmp_path: Path) -> None:
    import numpy as np

    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    sorter_dir = tmp_path / "sorter_output"
    sorter_dir.mkdir(parents=True, exist_ok=True)
    np.save(sorter_dir / "spike_times.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(sorter_dir / "spike_clusters.npy", np.array([1, 2, 210, 210], dtype=np.int32))

    class _ReadKiloSorting:
        def get_unit_ids(self):
            return [1, 2]

        def get_sampling_frequency(self):
            return 10000.0

    class _FullSorting:
        def __init__(self, unit_ids):
            self._unit_ids = list(unit_ids)

        def get_unit_ids(self):
            return list(self._unit_ids)

    calls: dict[str, object] = {}

    class _FakeNumpySorting:
        @staticmethod
        def from_times_labels(*, times_list, labels_list, sampling_frequency, unit_ids=None):
            calls["from_times_labels"] = {
                "sampling_frequency": float(sampling_frequency),
                "unit_ids": list(unit_ids or []),
                "n_times": int(len(times_list[0])),
                "n_labels": int(len(labels_list[0])),
            }
            return _FullSorting(unit_ids or [])

    class _FakeSI:
        NumpySorting = _FakeNumpySorting

        def read_kilosort(self, folder, keep_good_only=False, remove_empty_units=False):
            calls["read_kilosort"] = {
                "folder": str(Path(folder).resolve()),
                "keep_good_only": bool(keep_good_only),
                "remove_empty_units": bool(remove_empty_units),
            }
            return _ReadKiloSorting()

        def read_sorter_folder(self, folder, sorter_name=None):
            raise AssertionError("read_sorter_folder should not be used for this kilosort path")

        def load_extractor(self, folder):
            raise AssertionError("load_extractor should not be used for this kilosort path")

    loaded = spikesort_runner._load_sorting_from_sorter_output_dir(
        si_module=_FakeSI(),
        sorter_output_dir=sorter_dir,
        sorter_name="kilosort2_5",
    )

    assert calls.get("read_kilosort") == {
        "folder": str(sorter_dir.resolve()),
        "keep_good_only": False,
        "remove_empty_units": False,
    }
    assert calls.get("from_times_labels") == {
        "sampling_frequency": 10000.0,
        "unit_ids": [1, 2, 210],
        "n_times": 4,
        "n_labels": 4,
    }
    assert loaded.get_unit_ids() == [1, 2, 210]


def test_run_spikesort_stage_propagates_logging_debug_plot_report_inputs(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    captured_legacy_inputs: dict[str, object] = {}

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    class _LegacyOutputs:
        def __init__(self, output_subdir_after_well: str) -> None:
            self.recording_dir = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
            self.sorter_output_dir = well_out_dir / output_subdir_after_well / "sorter_output"
            self.output_dir = well_out_dir / output_subdir_after_well
            self.analyzer_dir = well_out_dir / output_subdir_after_well / "analyzer_output"
            self.merged_sorting_dir = None
            self.merged_sorter_output_dir = None

    def _fake_run_legacy_spikesorting_stage(*, inputs, logger):
        captured_legacy_inputs.update(
            {
                "preprocess_concat_recording_relpath": getattr(inputs, "preprocess_concat_recording_relpath", None),
                "log_enabled": bool(getattr(inputs, "log_enabled")),
                "log_verbose": bool(getattr(inputs, "log_verbose")),
                "log_file_override": getattr(inputs, "log_file_override"),
                "output_subdir_after_well": getattr(inputs, "output_subdir_after_well"),
                "plot_mode": getattr(inputs, "plot_mode"),
                "plot_debug": bool(getattr(inputs, "plot_debug")),
                "raster_sort": getattr(inputs, "raster_sort"),
                "fixed_y": bool(getattr(inputs, "fixed_y")),
                "run_reports": bool(getattr(inputs, "run_reports")),
                "no_curation": bool(getattr(inputs, "no_curation")),
                "export_to_phy": bool(getattr(inputs, "export_to_phy")),
            }
        )
        return _LegacyOutputs(str(getattr(inputs, "output_subdir_after_well")))

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fake_run_legacy_spikesorting_stage)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs_v2",
        preprocess_concat_recording_relpath="preprocess_outputs/preprocessed_recording",
        logging_enabled=False,
        logging_verbose=True,
        logging_file_relpath="logs/custom_spikesort.log",
        run_reports=False,
        plot_mode="merged",
        plot_debug=True,
        raster_sort="unit_id",
        fixed_y=True,
        no_curation=True,
        export_to_phy=True,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    assert captured_legacy_inputs.get("log_enabled") is False
    assert captured_legacy_inputs.get("log_verbose") is True
    assert captured_legacy_inputs.get("log_file_override") == "logs/custom_spikesort.log"
    assert captured_legacy_inputs.get("preprocess_concat_recording_relpath") == "preprocess_outputs/preprocessed_recording"
    assert captured_legacy_inputs.get("output_subdir_after_well") == "spikesort_outputs_v2"
    assert captured_legacy_inputs.get("plot_mode") == "merged"
    assert captured_legacy_inputs.get("plot_debug") is True
    assert captured_legacy_inputs.get("raster_sort") == "unit_id"
    assert captured_legacy_inputs.get("fixed_y") is True
    assert captured_legacy_inputs.get("run_reports") is False
    assert captured_legacy_inputs.get("no_curation") is True
    assert captured_legacy_inputs.get("export_to_phy") is True

    assert summary.get("inputs", {}).get("logging_enabled") is False
    assert summary.get("inputs", {}).get("logging_verbose") is True
    assert summary.get("inputs", {}).get("logging_file_relpath") == "logs/custom_spikesort.log"
    assert summary.get("inputs", {}).get("preprocess_concat_recording_relpath") == "preprocess_outputs/preprocessed_recording"
    assert summary.get("inputs", {}).get("plot_mode") == "merged"
    assert summary.get("inputs", {}).get("plot_debug") is True
    assert summary.get("inputs", {}).get("raster_sort") == "unit_id"
    assert summary.get("inputs", {}).get("fixed_y") is True
    assert summary.get("inputs", {}).get("run_reports") is False
    assert summary.get("inputs", {}).get("no_curation") is True
    assert summary.get("inputs", {}).get("export_to_phy") is True
    assert summary.get("inputs", {}).get("sort_enabled") is True
    assert summary.get("inputs", {}).get("sort_delete_outputs_on_force_restart") is False
    assert result.spikesort_out_dir == well_out_dir / "spikesort_outputs_v2"


def test_run_spikesort_stage_deletes_sort_outputs_on_force_restart_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    stage_out_dir = well_out_dir / "spikesort_outputs"
    sorter_out_dir = stage_out_dir / "sorter_output"
    analyzer_out_dir = stage_out_dir / "analyzer_output"
    sorter_out_dir.mkdir(parents=True, exist_ok=True)
    analyzer_out_dir.mkdir(parents=True, exist_ok=True)
    (sorter_out_dir / "stale.txt").write_text("old", encoding="utf-8")
    (analyzer_out_dir / "stale.txt").write_text("old", encoding="utf-8")

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    class _LegacyOutputs:
        def __init__(self) -> None:
            self.recording_dir = well_out_dir / "preprocess_outputs" / "preprocessed_recording"
            self.sorter_output_dir = stage_out_dir / "sorter_output"
            self.output_dir = stage_out_dir
            self.analyzer_dir = stage_out_dir / "analyzer_output"
            self.merged_sorting_dir = None
            self.merged_sorter_output_dir = None

    def _fake_run_legacy_spikesorting_stage(*, inputs, logger):
        assert not sorter_out_dir.exists()
        assert not analyzer_out_dir.exists()
        return _LegacyOutputs()

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _fake_run_legacy_spikesorting_stage)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        force_restart=True,
        sort_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    removed = summary.get("cleanup", {}).get("removed_on_force_restart", [])
    assert any(path.endswith("/spikesort_outputs/sorter_output") for path in removed)
    assert any(path.endswith("/spikesort_outputs/analyzer_output") for path in removed)


def test_run_spikesort_stage_skips_when_sort_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"

    def _fake_compute_mea_analysis_output_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
        return well_out_dir

    def _never_call_legacy(*, inputs, logger):
        raise AssertionError("legacy spikesort should not run when sort is disabled")

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", _fake_compute_mea_analysis_output_dir)
    monkeypatch.setattr(spikesort_runner, "run_legacy_spikesorting_stage", _never_call_legacy)

    inputs = SpikesortInputs(
        h5_path=tmp_path / "input.raw.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        sort_enabled=False,
    )

    result = run_spikesort_stage(inputs)
    summary = _read_json(result.summary_json)

    assert summary.get("status") == "skipped"
    assert summary.get("reason") == "sort_disabled"
    assert result.spikesort_out_dir == well_out_dir / "spikesort_outputs"
    assert result.spikesort_out_dir.exists()


def test_run_spikesort_merge_stage_writes_recommended_candidate_outputs(tmp_path: Path, monkeypatch) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    ks_wrapper_dir = well_out_dir / output_rel_root / "sorter_output"
    ks_dir = ks_wrapper_dir / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )

    stale_merge_out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
    stale_merge_out_dir.mkdir(parents=True, exist_ok=True)
    (stale_merge_out_dir / "stale.txt").write_text("old", encoding="utf-8")

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        assert package_root == "/tmp/slay"
        assert allow_numpy_fallback is True

        def _fake_run_slay(args):
            assert str(args["KS_folder"]) == str(ks_dir)
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(
                json.dumps({"100": [1, 2, 3]}),
                encoding="utf-8",
            )
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n"
                "1\t3\t0.92\t0.12\t0.02\t0.74\n"
                "2\t3\t0.93\t0.13\t0.03\t0.75\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 1}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params={"max_spikes": 123},
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=True,
    )

    summary = _read_json(result.summary_json)
    groups = _read_json(result.merge_out_dir / "recommended_merge_groups.json")
    candidates_tsv = (result.merge_out_dir / "recommended_merge_candidates.tsv").read_text(encoding="utf-8")

    assert result.merge_out_dir == well_out_dir / output_rel_root / "SLAy_outputs"
    assert summary.get("status") == "ok"
    assert summary.get("stage_output_root_dir") == str(well_out_dir / output_rel_root)
    assert summary.get("n_merge_groups") == 1
    assert summary.get("n_candidate_pairs") == 3
    slay_method = next((m for m in summary.get("methods", []) if str(m.get("name")) == "slay"), {})
    assert str(stale_merge_out_dir) in list(slay_method.get("removed_on_force_restart", []))
    assert not (stale_merge_out_dir / "stale.txt").exists()
    assert groups.get("n_groups") == 1
    assert groups.get("merge_groups", {}).get("100") == [1, 2, 3]
    assert "cluster_a\tcluster_b" in candidates_tsv
    assert "1\t2" in candidates_tsv
    assert "1\t3" in candidates_tsv
    assert "2\t3" in candidates_tsv


def test_run_spikesort_merge_stage_reports_plot_generation_note_when_auto_accept_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    ks_dir = well_out_dir / output_rel_root / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(
                json.dumps({"100": [1, 2]}),
                encoding="utf-8",
            )
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 1}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=True,
        slay_auto_accept_merges=True,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    slay_summary = _read_json(result.merge_out_dir / "slay_method_summary.json")

    assert slay_summary.get("plot_files_generated") == 0
    assert slay_summary.get("plot_files_generated_in_snapshot") == 0
    assert "plot_generation_note" in slay_summary
    assert "auto_accept_merges=true" in str(slay_summary.get("plot_generation_note", ""))



def test_run_spikesort_merge_stage_skips_when_disabled(tmp_path: Path) -> None:
    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    stage_cfg = SimpleNamespace(
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
        slay_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    assert result.merge_out_dir == result.well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    assert summary.get("status") == "skipped"
    assert summary.get("reason") == "slay_disabled"
    assert summary.get("stage_output_root_dir") == str(result.well_out_dir / "spikesort_outputs")


def test_run_spikesort_merge_stage_uses_merge_rel_output_root_for_stage_outputs_when_disabled(tmp_path: Path) -> None:
    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    stage_cfg = SimpleNamespace(
        merge_units_enabled=False,
        merge_rel_output_root="merge_outputs",
        slay_relpath="SLAy_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    assert result.merge_out_dir == result.well_out_dir / "spikesort_outputs" / "merge_outputs"
    assert summary.get("merge_rel_output_root") == "merge_outputs"
    assert summary.get("merge_output_rel_root") == "spikesort_outputs/merge_outputs"


def test_run_slay_merge_method_writes_outputs_under_merge_rel_output_root(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({"100": [1, 2]}), encoding="utf-8")
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n"
                "1\t2\t0.91\t0.11\t0.01\t0.73\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 1}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        merge_rel_output_root="merge_outputs",
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    expected_out_dir = (well_out_dir / output_rel_root / "merge_outputs" / "SLAy_outputs").resolve()
    assert Path(str(report.get("out_dir"))).resolve() == expected_out_dir
    assert (expected_out_dir / "slay_method_summary.json").exists()


def test_run_slay_merge_method_normalizes_wrapper_sorter_output_path(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    wrapper_dir = stage_output_root_dir / "sorter_output"
    ks_dir = wrapper_dir / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=False,
        slay_delete_outputs_on_force_restart=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=wrapper_dir,
    )

    assert Path(str(report.get("ks_dir"))).resolve() == ks_dir.resolve()


def test_run_slay_merge_method_disables_model_cache_read_and_write_when_knobs_false(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    captured_args: dict[str, object] = {}

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            captured_args.update(dict(args))
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 0}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_model_cache_relpath="cache/slay_model/ae.pt",
        slay_model_cache_use_cached_model=False,
        slay_model_cache_write_model=False,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(Path(str(report.get("summary_json"))))
    assert "model_path" not in captured_args
    assert summary.get("run_args", {}).get("model_path") is None
    assert summary.get("slay_model_cache_use_cached_model") is False
    assert summary.get("slay_model_cache_write_model") is False
    assert "slay.model_cache_path" not in dict(report.get("outputs", {}))


def test_run_slay_merge_method_retrains_without_using_existing_cached_model_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    output_rel_root = "spikesort_outputs"
    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / output_rel_root
    ks_dir = stage_output_root_dir / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    model_cache_path = stage_output_root_dir / "cache" / "slay_model" / "ae.pt"
    model_cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_cache_path.write_text("old-model", encoding="utf-8")

    captured_args: dict[str, object] = {}

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            captured_args.update(dict(args))
            model_path = Path(str(args["model_path"]))
            assert not model_path.exists()
            model_path.write_text("new-model", encoding="utf-8")

            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 0}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=True,
        slay_model_cache_relpath="cache/slay_model/ae.pt",
        slay_model_cache_use_cached_model=False,
        slay_model_cache_write_model=True,
        slay_params=None,
    )

    report = spikesort_runner._run_slay_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(Path(str(report.get("summary_json"))))
    assert str(captured_args.get("model_path")) == str(model_cache_path.resolve())
    assert model_cache_path.read_text(encoding="utf-8") == "new-model"
    assert summary.get("slay_model_deleted_to_disable_cache_use") is True
    assert summary.get("slay_model_cache_use_cached_model") is False
    assert summary.get("slay_model_cache_write_model") is True


def test_resolve_sorter_output_dir_resolves_slay_sorter_relpath_from_merge_root(tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    well_out_dir = tmp_path / "well001"
    ks_dir = well_out_dir / "spikesort_outputs" / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text("n_channels_dat=4\n", encoding="utf-8")

    resolved = spikesort_runner._resolve_sorter_output_dir(
        well_out_dir=well_out_dir,
        output_rel_root="spikesort_outputs",
        stage_config=SimpleNamespace(
            merge_rel_output_root="merge_outputs",
            slay_sorter_output_relpath="../sorter_output/sorter_output",
        ),
    )

    assert resolved == ks_dir.resolve()


def test_run_spikesort_merge_stage_skips_entire_phase_when_merge_units_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _never_run_slay(**kwargs):
        raise AssertionError("SLAy method should not execute when merge_units is disabled")

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _never_run_slay)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=False,
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "skipped"
    assert summary.get("reason") == "merge_units_disabled"
    assert summary.get("methods") == []
    assert summary.get("merge_units_enabled") is False
    assert result.outputs.get("summary_json") == str(result.summary_json)


def test_run_spikesort_merge_stage_caches_sorting_outputs_before_merge_when_enabled(tmp_path: Path) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    analyzer_output_dir = stage_output_root_dir / "analyzer_output"
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    (sorter_output_dir / "sorter_marker.txt").write_text("sorter", encoding="utf-8")
    (analyzer_output_dir / "analyzer_marker.txt").write_text("analyzer", encoding="utf-8")

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        cache_sorting_outputs_before_merge=True,
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
        slay_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    cache_root_dir = (stage_output_root_dir / "pre_merge_cache").resolve()

    assert summary.get("cache_sorting_outputs_before_merge") is True
    assert result.outputs.get("merge.pre_merge_cache_dir") == str(cache_root_dir)
    assert result.outputs.get("merge.pre_merge_cache_summary_json") == str(cache_root_dir / "pre_merge_cache_summary.json")
    assert (cache_root_dir / "sorter_output" / "sorter_marker.txt").read_text(encoding="utf-8") == "sorter"
    assert (cache_root_dir / "analyzer_output" / "analyzer_marker.txt").read_text(encoding="utf-8") == "analyzer"


def test_run_spikesort_merge_stage_caches_outputs_under_merge_rel_output_root(tmp_path: Path) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    analyzer_output_dir = stage_output_root_dir / "analyzer_output"
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    (sorter_output_dir / "sorter_marker.txt").write_text("sorter", encoding="utf-8")
    (analyzer_output_dir / "analyzer_marker.txt").write_text("analyzer", encoding="utf-8")

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        merge_rel_output_root="merge_outputs",
        cache_sorting_outputs_before_merge=True,
        cache_sorting_outputs_before_merge_relpath="cache/pre_merge_cache",
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
        slay_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    expected_cache_root = (stage_output_root_dir / "merge_outputs" / "cache" / "pre_merge_cache").resolve()

    assert result.outputs.get("merge.pre_merge_cache_dir") == str(expected_cache_root)
    assert result.outputs.get("merge.pre_merge_cache_summary_json") == str(
        expected_cache_root / "pre_merge_cache_summary.json"
    )
    assert (expected_cache_root / "sorter_output" / "sorter_marker.txt").read_text(encoding="utf-8") == "sorter"
    assert (expected_cache_root / "analyzer_output" / "analyzer_marker.txt").read_text(encoding="utf-8") == "analyzer"


def test_run_spikesort_merge_stage_uses_existing_cache_on_force_restart_when_enabled(tmp_path: Path) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    analyzer_output_dir = stage_output_root_dir / "analyzer_output"
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    (sorter_output_dir / "sorter_marker.txt").write_text("current", encoding="utf-8")
    (analyzer_output_dir / "analyzer_marker.txt").write_text("current", encoding="utf-8")

    cache_root_dir = stage_output_root_dir / "cache" / "pre_merge_cache"
    cache_sorter_dir = cache_root_dir / "sorter_output"
    cache_analyzer_dir = cache_root_dir / "analyzer_output"
    cache_sorter_dir.mkdir(parents=True, exist_ok=True)
    cache_analyzer_dir.mkdir(parents=True, exist_ok=True)
    (cache_sorter_dir / "sorter_marker.txt").write_text("cached", encoding="utf-8")
    (cache_analyzer_dir / "analyzer_marker.txt").write_text("cached", encoding="utf-8")

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        cache_sorting_outputs_before_merge=True,
        cache_sorting_outputs_before_merge_relpath="cache/pre_merge_cache",
        cache_sorting_outputs_before_merge_cleanup_on_success=False,
        cache_sorting_outputs_before_merge_use_cache_on_force_restart=True,
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
        slay_delete_outputs_on_force_restart=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=True,
    )

    summary = _read_json(result.summary_json)

    assert (sorter_output_dir / "sorter_marker.txt").read_text(encoding="utf-8") == "cached"
    assert (analyzer_output_dir / "analyzer_marker.txt").read_text(encoding="utf-8") == "cached"
    assert summary.get("cache_sorting_outputs_before_merge_config", {}).get("restored_from_existing_cache") is True
    assert (
        summary.get("cache_sorting_outputs_before_merge_config", {}).get("replace_sorting_with_cache_before_force_restart")
        is True
    )
    assert result.outputs.get("merge.pre_merge_cache_dir") == str(cache_root_dir.resolve())


def test_run_spikesort_merge_stage_cleans_up_cache_on_success_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    sorter_output_dir = stage_output_root_dir / "sorter_output"
    analyzer_output_dir = stage_output_root_dir / "analyzer_output"
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    (sorter_output_dir / "sorter_marker.txt").write_text("sorter", encoding="utf-8")
    (analyzer_output_dir / "analyzer_marker.txt").write_text("analyzer", encoding="utf-8")

    def _fake_slay(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart):
        out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "slay_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.summary_json": str(summary_json)},
            "ks_dir": str(out_dir / "sorter_output"),
            "applied_merges": False,
            "n_merge_groups": 0,
            "n_candidate_pairs": 0,
        }

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        cache_sorting_outputs_before_merge=True,
        cache_sorting_outputs_before_merge_relpath="cache/pre_merge_cache",
        cache_sorting_outputs_before_merge_cleanup_on_success=True,
        cache_sorting_outputs_before_merge_use_cache_on_force_restart=False,
        merge_sequence=("SLAy",),
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    cache_root_dir = (stage_output_root_dir / "cache" / "pre_merge_cache").resolve()

    assert summary.get("status") == "ok"
    assert summary.get("cache_sorting_outputs_before_merge_config", {}).get("cleaned_up") is True
    assert not cache_root_dir.exists()
    assert result.outputs.get("merge.pre_merge_cache_dir") is None


def test_run_spikesort_merge_stage_runs_methods_in_canonical_workspace_without_publish(
    tmp_path: Path, monkeypatch
) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    live_sorter_dir = stage_output_root_dir / "sorter_output"
    live_analyzer_dir = stage_output_root_dir / "analyzer_output"
    live_sorter_dir.mkdir(parents=True, exist_ok=True)
    live_analyzer_dir.mkdir(parents=True, exist_ok=True)
    (live_sorter_dir / "sorter_marker.txt").write_text("live", encoding="utf-8")
    (live_analyzer_dir / "analyzer_marker.txt").write_text("live", encoding="utf-8")

    seen_stage_output_roots: list[Path] = []

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        seen_stage_output_roots.append(Path(stage_output_root_dir).resolve())
        marker = Path(stage_output_root_dir) / "sorter_output" / "sorter_marker.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("canonical-merged", encoding="utf-8")
        out_dir = Path(stage_output_root_dir) / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        merge_rel_output_root="merge_outputs",
        merge_sequence=("auto_merge",),
        cache_sorting_outputs_before_merge_use_canonical_workspace=True,
        cache_sorting_outputs_before_merge_canonical_workspace_relpath="cache/merge_workspace",
        cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=True,
        cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=False,
        auto_merge_enabled=True,
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    canonical_root = (stage_output_root_dir / "merge_outputs" / "cache" / "merge_workspace").resolve()

    assert seen_stage_output_roots == [canonical_root]
    assert (live_sorter_dir / "sorter_marker.txt").read_text(encoding="utf-8") == "live"
    assert (canonical_root / "sorter_output" / "sorter_marker.txt").read_text(encoding="utf-8") == "canonical-merged"
    assert summary.get("cache_sorting_outputs_before_merge_config", {}).get("canonical_workspace_prepared") is True
    assert summary.get("cache_sorting_outputs_before_merge_config", {}).get("canonical_workspace_published") is False
    assert result.outputs.get("merge.canonical_workspace_dir") == str(canonical_root)


def test_run_spikesort_merge_stage_asserts_slay_uses_canonical_workspace_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    live_sorter_dir = stage_output_root_dir / "sorter_output"
    live_analyzer_dir = stage_output_root_dir / "analyzer_output"
    live_sorter_dir.mkdir(parents=True, exist_ok=True)
    live_analyzer_dir.mkdir(parents=True, exist_ok=True)
    (live_sorter_dir / "sorter_marker.txt").write_text("live", encoding="utf-8")
    (live_analyzer_dir / "analyzer_marker.txt").write_text("live", encoding="utf-8")

    def _fake_slay(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir=None):
        out_dir = Path(stage_output_root_dir) / "SLAy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "slay_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.summary_json": str(summary_json)},
            "ks_dir": str(live_sorter_dir.resolve()),
            "applied_merges": False,
        }

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        merge_rel_output_root="merge_outputs",
        merge_sequence=("SLAy",),
        cache_sorting_outputs_before_merge_use_canonical_workspace=True,
        cache_sorting_outputs_before_merge_canonical_workspace_relpath="cache/merge_workspace",
        cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=True,
        cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=False,
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
    )

    with pytest.raises(RuntimeError, match="SLAy expected canonical workspace sorter output"):
        run_spikesort_merge_stage(
            h5_path=h5_path,
            stream_id=stream_id,
            mea_output_root=tmp_path,
            output_rel_root=output_rel_root,
            stage_config=stage_cfg,
            force_restart=False,
        )


def test_run_spikesort_merge_stage_asserts_auto_merge_uses_canonical_workspace_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    live_sorter_dir = stage_output_root_dir / "sorter_output"
    live_analyzer_dir = stage_output_root_dir / "analyzer_output"
    live_sorter_dir.mkdir(parents=True, exist_ok=True)
    live_analyzer_dir.mkdir(parents=True, exist_ok=True)
    (live_sorter_dir / "sorter_marker.txt").write_text("live", encoding="utf-8")
    (live_analyzer_dir / "analyzer_marker.txt").write_text("live", encoding="utf-8")

    def _fake_slay(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir=None):
        out_dir = Path(stage_output_root_dir) / "SLAy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "slay_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.summary_json": str(summary_json)},
            "ks_dir": str(live_sorter_dir.resolve()),
            "applied_merges": False,
        }

    def _auto_merge_should_not_run(**kwargs):
        raise AssertionError("auto_merge should not run when canonical assertion fails")

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _auto_merge_should_not_run)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        merge_rel_output_root="merge_outputs",
        merge_sequence=("SLAy", "auto_merge"),
        cache_sorting_outputs_before_merge_use_canonical_workspace=True,
        cache_sorting_outputs_before_merge_canonical_workspace_relpath="cache/merge_workspace",
        cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=True,
        cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=False,
        cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace=False,
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        auto_merge_enabled=True,
    )

    with pytest.raises(RuntimeError, match="auto_merge expected canonical workspace sorter output"):
        run_spikesort_merge_stage(
            h5_path=h5_path,
            stream_id=stream_id,
            mea_output_root=tmp_path,
            output_rel_root=output_rel_root,
            stage_config=stage_cfg,
            force_restart=False,
        )


def test_run_spikesort_merge_stage_publishes_canonical_workspace_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    stage_output_root_dir = well_out_dir / output_rel_root
    live_sorter_dir = stage_output_root_dir / "sorter_output"
    live_analyzer_dir = stage_output_root_dir / "analyzer_output"
    live_sorter_dir.mkdir(parents=True, exist_ok=True)
    live_analyzer_dir.mkdir(parents=True, exist_ok=True)
    (live_sorter_dir / "sorter_marker.txt").write_text("live", encoding="utf-8")
    (live_analyzer_dir / "analyzer_marker.txt").write_text("live", encoding="utf-8")

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        marker = Path(stage_output_root_dir) / "sorter_output" / "sorter_marker.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("canonical-merged", encoding="utf-8")
        out_dir = Path(stage_output_root_dir) / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)

    stage_cfg = SimpleNamespace(
        merge_units_enabled=True,
        merge_rel_output_root="merge_outputs",
        merge_sequence=("auto_merge",),
        cache_sorting_outputs_before_merge_use_canonical_workspace=True,
        cache_sorting_outputs_before_merge_canonical_workspace_relpath="cache/merge_workspace",
        cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=True,
        cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=False,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=True,
        cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=False,
        auto_merge_enabled=True,
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)

    assert (live_sorter_dir / "sorter_marker.txt").read_text(encoding="utf-8") == "canonical-merged"
    assert summary.get("cache_sorting_outputs_before_merge_config", {}).get("canonical_workspace_prepared") is True
    assert summary.get("cache_sorting_outputs_before_merge_config", {}).get("canonical_workspace_published") is True
    assert result.outputs.get("merge.published_sorter_output_dir") == str(live_sorter_dir.resolve())


def test_run_spikesort_merge_stage_preserves_existing_outputs_when_delete_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    output_rel_root = "spikesort_outputs"
    stream_id = "well001"
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=tmp_path,
        data_file=h5_path,
        well=stream_id,
    )
    ks_dir = well_out_dir / output_rel_root / "sorter_output" / "sorter_output"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "params.py").write_text(
        "dat_path = 'data.bin'\n"
        "n_channels_dat = 4\n"
        "dtype = 'int16'\n"
        "sample_rate = 30000\n",
        encoding="utf-8",
    )

    merge_out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = merge_out_dir / "keep_me.txt"
    sentinel.write_text("persist", encoding="utf-8")

    def _fake_import_slay_run_function(*, package_root, allow_numpy_fallback):
        def _fake_run_slay(args):
            automerge_dir = Path(args["KS_folder"]) / "automerge"
            automerge_dir.mkdir(parents=True, exist_ok=True)
            (automerge_dir / "new2old.json").write_text(json.dumps({}), encoding="utf-8")
            (automerge_dir / "metrics.tsv").write_text(
                "Cluster 1\tCluster 2\tSimilarity\tCross-correlation Significance\tRefractory Period Penalty\tFinal Metric\n",
                encoding="utf-8",
            )
            Path(args["output_json"]).write_text(json.dumps({"num_merges": 0}), encoding="utf-8")

        return _fake_run_slay

    monkeypatch.setattr(spikesort_runner, "_import_slay_run_function", _fake_import_slay_run_function)

    stage_cfg = SimpleNamespace(
        slay_enabled=True,
        slay_relpath="SLAy_outputs",
        slay_package_root="/tmp/slay",
        slay_sorter_output_relpath=None,
        slay_output_json_relpath="run-output.json",
        slay_candidate_pairs_relpath="recommended_merge_candidates.tsv",
        slay_merge_groups_relpath="recommended_merge_groups.json",
        slay_allow_numpy_fallback=True,
        slay_plot_merges=False,
        slay_auto_accept_merges=False,
        slay_copy_automerge_artifacts=True,
        slay_delete_outputs_on_force_restart=False,
        slay_params={"max_spikes": 10},
    )

    run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=tmp_path,
        output_rel_root=output_rel_root,
        stage_config=stage_cfg,
        force_restart=True,
    )

    assert sentinel.exists()


def test_run_spikesort_merge_stage_sequences_methods_and_recomputes_after_slay(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    order: list[str] = []

    def _fake_slay(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart):
        order.append("slay")
        out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "slay_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.summary_json": str(summary_json)},
            "ks_dir": str(out_dir / "sorter_output"),
            "applied_merges": True,
            "n_merge_groups": 1,
            "n_candidate_pairs": 1,
        }

    def _fake_recompute(*, well_out_dir, stage_output_root_dir, stage_config, sorter_output_dir):
        order.append("recompute")
        summary_json = stage_output_root_dir / "slay_analyzer_recompute_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay_recompute_analyzer",
            "status": "ok",
            "reason": None,
            "out_dir": str(stage_output_root_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.recompute_analyzer.summary_json": str(summary_json)},
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        order.append("auto_merge")
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 2,
            "n_candidate_pairs_total": 3,
            "n_applied_groups_total": 1,
            "n_iterations": 2,
        }

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)
    monkeypatch.setattr(spikesort_runner, "_run_slay_analyzer_recompute", _fake_recompute)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy", "auto_merge"),
        slay_recompute_analyzer=True,
        slay_auto_accept_merges=True,
        slay_relpath="SLAy_outputs",
        auto_merge_relpath="automerge_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    method_names = [str(m.get("name")) for m in summary.get("methods", [])]

    assert order == ["slay", "recompute", "auto_merge"]
    assert method_names == ["slay", "slay_recompute_analyzer", "auto_merge"]
    assert summary.get("status") == "ok"
    assert summary.get("auto_merge_n_iterations") == 2


def test_run_spikesort_merge_stage_recomputes_after_slay_without_pending_auto_merge(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    order: list[str] = []

    def _fake_slay(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart):
        order.append("slay")
        out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "slay_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.summary_json": str(summary_json)},
            "ks_dir": str(out_dir / "sorter_output"),
            "applied_merges": True,
            "n_merge_groups": 1,
            "n_candidate_pairs": 1,
        }

    def _fake_recompute(*, well_out_dir, stage_output_root_dir, stage_config, sorter_output_dir):
        order.append("recompute")
        summary_json = stage_output_root_dir / "slay_analyzer_recompute_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay_recompute_analyzer",
            "status": "ok",
            "reason": None,
            "out_dir": str(stage_output_root_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.recompute_analyzer.summary_json": str(summary_json)},
        }

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)
    monkeypatch.setattr(spikesort_runner, "_run_slay_analyzer_recompute", _fake_recompute)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy",),
        slay_recompute_analyzer=True,
        slay_auto_accept_merges=True,
        slay_relpath="SLAy_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    method_names = [str(m.get("name")) for m in summary.get("methods", [])]

    assert order == ["slay", "recompute"]
    assert method_names == ["slay", "slay_recompute_analyzer"]
    assert summary.get("status") == "ok"


def test_run_spikesort_merge_stage_does_not_recompute_when_slay_auto_accept_disabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    order: list[str] = []

    def _fake_slay(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart):
        order.append("slay")
        out_dir = well_out_dir / output_rel_root / "SLAy_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "slay_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.summary_json": str(summary_json)},
            "ks_dir": str(out_dir / "sorter_output"),
            "applied_merges": False,
            "n_merge_groups": 0,
            "n_candidate_pairs": 0,
        }

    def _fake_recompute(*, well_out_dir, stage_output_root_dir, stage_config, sorter_output_dir):
        order.append("recompute")
        summary_json = stage_output_root_dir / "slay_analyzer_recompute_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "slay_recompute_analyzer",
            "status": "ok",
            "reason": None,
            "out_dir": str(stage_output_root_dir),
            "summary_json": str(summary_json),
            "outputs": {"slay.recompute_analyzer.summary_json": str(summary_json)},
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        order.append("auto_merge")
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 0,
            "n_candidate_pairs_total": 0,
            "n_applied_groups_total": 0,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fake_slay)
    monkeypatch.setattr(spikesort_runner, "_run_slay_analyzer_recompute", _fake_recompute)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy", "auto_merge"),
        slay_recompute_analyzer=True,
        slay_auto_accept_merges=False,
        slay_relpath="SLAy_outputs",
        auto_merge_relpath="automerge_outputs",
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    method_names = [str(m.get("name")) for m in summary.get("methods", [])]

    assert order == ["slay", "auto_merge"]
    assert method_names == ["slay", "auto_merge"]


def test_run_spikesort_merge_stage_writes_single_merge_metadata_summary_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    capture_calls: list[dict[str, object]] = []

    def _fake_capture(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, capture_label, include_unit_locations, allow_analyzer_recompute):
        capture_calls.append(
            {
                "capture_label": capture_label,
                "allow_analyzer_recompute": bool(allow_analyzer_recompute),
                "include_unit_locations": bool(include_unit_locations),
            }
        )
        if capture_label == "before_merge":
            return {
                "sorter": {
                    "available": True,
                    "unit_ids": ["1", "2", "3"],
                },
                "analyzer": {
                    "available": True,
                    "unit_ids": ["1", "2", "3"],
                    "unit_locations_by_unit": {
                        "1": {"x_um": 10.0, "y_um": 20.0},
                        "2": {"x_um": 20.0, "y_um": 30.0},
                        "3": {"x_um": 30.0, "y_um": 40.0},
                    },
                },
            }
        return {
            "sorter": {
                "available": True,
                "unit_ids": ["1", "2", "3"],
            },
            "analyzer": {
                "available": True,
                "unit_ids": ["1", "3"],
                "unit_locations_by_unit": {
                    "1": {"x_um": 11.0, "y_um": 21.0},
                    "3": {"x_um": 30.0, "y_um": 40.0},
                },
            },
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)
    monkeypatch.setattr(
        spikesort_runner,
        "_extract_applied_merge_operations",
        lambda *, method_reports, stage_config: [
            {
                "method": "auto_merge",
                "iteration": 1,
                "template_diff_thresh": 0.05,
                "group_id": "auto_merge_iter_001_group_001",
                "pre_unit_ids": ["1", "2"],
            }
        ],
    )

    stage_cfg = SimpleNamespace(
        merge_sequence=("auto_merge",),
        slay_enabled=False,
        slay_auto_accept_merges=False,
        slay_relpath="SLAy_outputs",
        auto_merge_enabled=True,
        auto_merge_auto_accept_merges=True,
        auto_merge_relpath="automerge_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_metadata_include_unit_locations=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    metadata_path = Path(str(result.outputs.get("merge.metadata_summary_json")))
    metadata = _read_json(metadata_path)

    assert metadata_path.exists()
    assert summary.get("merge_metadata_summary_json") == str(metadata_path)
    assert metadata.get("applied_merge_group_count") == 1
    assert metadata.get("delta", {}).get("any_changed") is True
    assert metadata.get("change_validation", {}).get("expected_change_if_auto_accept_enabled") is True
    assert metadata.get("change_validation", {}).get("expected_change_if_merges_applied") is True
    assert metadata.get("change_validation", {}).get("passes") is True
    assert capture_calls == [
        {
            "capture_label": "before_merge",
            "allow_analyzer_recompute": True,
            "include_unit_locations": True,
        },
        {
            "capture_label": "after_merge",
            "allow_analyzer_recompute": True,
            "include_unit_locations": True,
        },
    ]


def test_run_spikesort_merge_stage_writes_pre_and_post_metadata_summaries_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _fake_capture(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, capture_label, include_unit_locations, allow_analyzer_recompute):
        if capture_label == "before_merge":
            return {
                "sorter": {
                    "available": True,
                    "unit_ids": ["1", "2", "3"],
                },
                "analyzer": {
                    "available": True,
                    "unit_ids": ["1", "2", "3"],
                    "unit_locations_by_unit": {},
                },
            }
        return {
            "sorter": {
                "available": True,
                "unit_ids": ["1", "3"],
            },
            "analyzer": {
                "available": True,
                "unit_ids": ["1", "3"],
                "unit_locations_by_unit": {},
            },
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 0,
            "n_candidate_pairs_total": 0,
            "n_applied_groups_total": 0,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)

    stage_cfg = SimpleNamespace(
        merge_sequence=("auto_merge",),
        merge_units_enabled=True,
        slay_enabled=False,
        slay_relpath="SLAy_outputs",
        auto_merge_enabled=True,
        auto_merge_auto_accept_merges=False,
        auto_merge_relpath="automerge_outputs",
        merge_metadata_enabled=False,
        pre_merge_metadata_enabled=True,
        pre_merge_metadata_write_json=True,
        pre_merge_metadata_json_relpath="pre_merge_metadata_summary.json",
        pre_merge_metadata_include_unit_locations=True,
        post_merge_metadata_enabled=True,
        post_merge_metadata_write_json=True,
        post_merge_metadata_json_relpath="post_merge_metadata_summary.json",
        post_merge_metadata_include_unit_locations=True,
        merge_reports_enabled=False,
        merge_reports_2panel_enabled=False,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    pre_path = Path(str(result.outputs.get("merge.pre_metadata_summary_json")))
    post_path = Path(str(result.outputs.get("merge.post_metadata_summary_json")))
    pre_payload = _read_json(pre_path)
    post_payload = _read_json(post_path)

    assert pre_path.exists()
    assert post_path.exists()
    assert summary.get("pre_merge_metadata_enabled") is True
    assert summary.get("post_merge_metadata_enabled") is True
    assert summary.get("pre_merge_metadata_summary_json") == str(pre_path)
    assert summary.get("post_merge_metadata_summary_json") == str(post_path)
    assert pre_payload.get("snapshot_label") == "before_merge"
    assert post_payload.get("snapshot_label") == "after_merge"
    assert pre_payload.get("summary", {}).get("analyzer", {}).get("unit_count") == 3
    assert post_payload.get("summary", {}).get("analyzer", {}).get("unit_count") == 2


def test_run_spikesort_merge_stage_merge_metadata_flags_no_change_when_auto_accept_applied(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _fake_capture(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, capture_label, include_unit_locations, allow_analyzer_recompute):
        return {
            "sorter": {
                "available": True,
                "unit_ids": ["1", "2"],
            },
            "analyzer": {
                "available": True,
                "unit_ids": ["1", "2"],
                "unit_locations_by_unit": {
                    "1": {"x_um": 10.0, "y_um": 20.0},
                    "2": {"x_um": 20.0, "y_um": 30.0},
                },
            },
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)
    monkeypatch.setattr(
        spikesort_runner,
        "_extract_applied_merge_operations",
        lambda *, method_reports, stage_config: [
            {
                "method": "auto_merge",
                "iteration": 1,
                "template_diff_thresh": 0.05,
                "group_id": "auto_merge_iter_001_group_001",
                "pre_unit_ids": ["1", "2"],
            }
        ],
    )

    stage_cfg = SimpleNamespace(
        merge_sequence=("auto_merge",),
        slay_enabled=False,
        slay_auto_accept_merges=False,
        slay_relpath="SLAy_outputs",
        auto_merge_enabled=True,
        auto_merge_auto_accept_merges=True,
        auto_merge_relpath="automerge_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_metadata_include_unit_locations=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    metadata_path = Path(str(result.outputs.get("merge.metadata_summary_json")))
    metadata = _read_json(metadata_path)

    assert metadata.get("delta", {}).get("any_changed") is False
    assert metadata.get("change_validation", {}).get("expected_change_if_auto_accept_enabled") is True
    assert metadata.get("change_validation", {}).get("expected_change_if_merges_applied") is True
    assert metadata.get("change_validation", {}).get("passes") is False
    assert (
        metadata.get("change_validation", {}).get("reason")
        == "auto_accept_enabled_and_merges_applied_but_no_before_after_unit_change_detected"
    )


def test_run_spikesort_merge_stage_logs_merge_summary_details_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _fake_capture(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, capture_label, include_unit_locations, allow_analyzer_recompute):
        if capture_label == "before_merge":
            return {
                "sorter": {
                    "available": True,
                    "unit_count": 3,
                    "unit_ids": ["1", "2", "3"],
                },
                "analyzer": {
                    "available": True,
                    "unit_count": 3,
                    "unit_ids": ["1", "2", "3"],
                    "unit_locations_by_unit": {},
                },
            }
        return {
            "sorter": {
                "available": True,
                "unit_count": 2,
                "unit_ids": ["1", "3"],
            },
            "analyzer": {
                "available": True,
                "unit_count": 2,
                "unit_ids": ["1", "3"],
                "unit_locations_by_unit": {},
            },
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)
    monkeypatch.setattr(
        spikesort_runner,
        "_extract_applied_merge_operations",
        lambda *, method_reports, stage_config: [
            {
                "method": "auto_merge",
                "iteration": 1,
                "template_diff_thresh": 0.05,
                "group_id": "auto_merge_iter_001_group_001",
                "pre_unit_ids": ["1", "2"],
            }
        ],
    )

    info_messages: list[str] = []
    warning_messages: list[str] = []

    def _capture_info(message: str, *args, **kwargs) -> None:
        info_messages.append((message % args) if args else str(message))

    def _capture_warning(message: str, *args, **kwargs) -> None:
        warning_messages.append((message % args) if args else str(message))

    monkeypatch.setattr(spikesort_runner.LOGGER, "info", _capture_info)
    monkeypatch.setattr(spikesort_runner.LOGGER, "warning", _capture_warning)

    stage_cfg = SimpleNamespace(
        merge_sequence=("auto_merge",),
        slay_enabled=False,
        slay_auto_accept_merges=False,
        slay_relpath="SLAy_outputs",
        auto_merge_enabled=True,
        auto_merge_auto_accept_merges=True,
        auto_merge_relpath="automerge_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_metadata_include_unit_locations=True,
        merge_metadata_log_summary_details=True,
    )

    run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    assert any("Merge summary [stream=well001]" in line for line in info_messages)
    assert any("Merge metadata [stream=well001]" in line for line in info_messages)
    assert warning_messages == []


def test_run_spikesort_merge_stage_writes_merge_reports_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _fake_capture(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, capture_label, include_unit_locations, allow_analyzer_recompute):
        if capture_label == "before_merge":
            return {
                "sorter": {
                    "available": True,
                    "unit_count": 3,
                    "unit_ids": ["1", "2", "3"],
                },
                "analyzer": {
                    "available": True,
                    "unit_count": 3,
                    "unit_ids": ["1", "2", "3"],
                    "unit_locations_by_unit": {
                        "1": {"x_um": 10.0, "y_um": 20.0},
                        "2": {"x_um": 30.0, "y_um": 40.0},
                        "3": {"x_um": 50.0, "y_um": 60.0},
                    },
                },
            }

        return {
            "sorter": {
                "available": True,
                "unit_count": 2,
                "unit_ids": ["1", "3"],
            },
            "analyzer": {
                "available": True,
                "unit_count": 2,
                "unit_ids": ["1", "3"],
                "unit_locations_by_unit": {
                    "1": {"x_um": 11.0, "y_um": 21.0},
                    "3": {"x_um": 51.0, "y_um": 61.0},
                },
            },
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "merge_out_dir": str(merge_out_dir),
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 3,
            "after_unit_locations_count": 2,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("auto_merge",),
        slay_enabled=False,
        auto_merge_enabled=True,
        auto_merge_relpath="automerge_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_include_unit_locations=True,
        merge_reports_enabled=True,
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)

    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 3
    assert report_calls[0].get("after_count") == 2
    assert report_calls[0].get("applied_mappings_count") == 0
    assert summary.get("merge_reports", {}).get("status") == "ok"
    assert summary.get("merge_reports", {}).get("before_unit_locations_count") == 3
    assert summary.get("merge_reports", {}).get("after_unit_locations_count") == 2
    assert "merge_reports_error" not in summary
    assert "merge.report.unit_locations_before_after_png" in result.outputs


def test_run_spikesort_merge_stage_writes_unit_diff_json_and_uses_it_for_2panel(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    def _fake_capture(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, capture_label, include_unit_locations, allow_analyzer_recompute):
        if capture_label == "before_merge":
            return {
                "sorter": {
                    "available": True,
                    "unit_ids": ["1", "2"],
                },
                "analyzer": {
                    "available": True,
                    "unit_ids": ["1", "2"],
                    "unit_locations_by_unit": {
                        "1": {"x_um": 10.0, "y_um": 20.0},
                        "2": {"x_um": 30.0, "y_um": 40.0},
                    },
                },
            }
        return {
            "sorter": {
                "available": True,
                "unit_ids": ["1"],
            },
            "analyzer": {
                "available": True,
                "unit_ids": ["1"],
                "unit_locations_by_unit": {
                    "1": {"x_um": 11.0, "y_um": 21.0},
                },
            },
        }

    def _fake_auto_merge(*, well_out_dir, stage_output_root_dir, output_rel_root, stage_config, force_restart, sorter_output_dir):
        out_dir = well_out_dir / output_rel_root / "automerge_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_json = out_dir / "auto_merge_method_summary.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {
            "name": "auto_merge",
            "status": "ok",
            "reason": None,
            "out_dir": str(out_dir),
            "summary_json": str(summary_json),
            "outputs": {"auto_merge.summary_json": str(summary_json)},
            "n_candidate_groups_total": 1,
            "n_candidate_pairs_total": 1,
            "n_applied_groups_total": 1,
            "n_iterations": 1,
        }

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 2,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "_capture_merge_state_snapshot", _fake_capture)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fake_auto_merge)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("auto_merge",),
        slay_enabled=False,
        auto_merge_enabled=True,
        auto_merge_relpath="automerge_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_include_unit_locations=True,
        merge_reports_enabled=True,
        merge_reports_unit_diff_json_enabled=True,
        merge_reports_unit_diff_json_relpath="unit_diffs_after_merge.json",
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
    )

    summary = _read_json(result.summary_json)
    unit_diff_json = Path(str(result.outputs.get("merge.report.unit_diff_json")))
    unit_diff_payload = _read_json(unit_diff_json)

    assert unit_diff_json.exists()
    assert summary.get("merge_unit_diff_json") == str(unit_diff_json)
    assert report_calls == [
        {
            "before_count": 2,
            "after_count": 1,
            "applied_mappings_count": 0,
        }
    ]
    assert len(unit_diff_payload.get("before", {}).get("analyzer", {}).get("unit_locations_by_unit", {})) == 2
    assert len(unit_diff_payload.get("after", {}).get("analyzer", {}).get("unit_locations_by_unit", {})) == 1


def test_run_spikesort_merge_stage_force_replot_uses_unit_diff_json_as_2panel_source_of_truth(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    merge_out_dir = well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_json = merge_out_dir / "merge_metadata_summary.json"
    metadata_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "9": {"x_um": 90.0, "y_um": 90.0},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "9": {"x_um": 91.0, "y_um": 91.0},
                        }
                    }
                },
                "applied_unit_mappings": [
                    {
                        "pre_unit_ids": ["9"],
                        "post_unit_id": "9",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    unit_diff_json = merge_out_dir / "unit_diffs_after_merge.json"
    unit_diff_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 10.0, "y_um": 20.0},
                            "2": {"x_um": 30.0, "y_um": 40.0},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 11.0, "y_um": 21.0},
                        }
                    }
                },
                "applied_unit_mappings": [
                    {
                        "pre_unit_ids": ["1", "2"],
                        "post_unit_id": "1",
                    }
                ],
                "applied_merge_group_count": 1,
            }
        ),
        encoding="utf-8",
    )

    summary_json = merge_out_dir / "merge_stage_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "methods": [
                    {"name": "slay", "status": "ok"},
                ],
                "merge_metadata_summary_json": str(metadata_json),
                "merge_unit_diff_json": str(unit_diff_json),
                "outputs": {
                    "merge.metadata_summary_json": str(metadata_json),
                    "merge.report.unit_diff_json": str(unit_diff_json),
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("merge methods must not run in force_replot-only mode")

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 2,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy",),
        merge_units_enabled=True,
        slay_relpath="SLAy_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_reports_enabled=True,
        merge_reports_unit_diff_json_enabled=True,
        merge_reports_unit_diff_json_relpath="unit_diffs_after_merge.json",
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=True,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "ok"
    assert summary.get("replot_only") is True
    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 2
    assert report_calls[0].get("after_count") == 1
    assert report_calls[0].get("applied_mappings_count") == 1
    assert result.outputs.get("merge.report.unit_diff_json") == str(unit_diff_json)


def test_run_spikesort_merge_stage_force_replot_only_uses_existing_metadata(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    merge_out_dir = well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_json = merge_out_dir / "merge_metadata_summary.json"
    metadata_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 10.0, "y_um": 20.0},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "1": {"x_um": 11.0, "y_um": 21.0},
                        }
                    }
                },
                "applied_unit_mappings": [
                    {
                        "pre_unit_ids": ["1"],
                        "post_unit_id": "1",
                    }
                ],
                "applied_merge_group_count": 1,
                "change_validation": {"passes": True},
            }
        ),
        encoding="utf-8",
    )

    summary_json = merge_out_dir / "merge_stage_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "methods": [
                    {"name": "slay", "status": "ok"},
                ],
                "merge_metadata_summary_json": str(metadata_json),
                "outputs": {
                    "merge.metadata_summary_json": str(metadata_json),
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("merge methods must not run in force_replot-only mode")

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 1,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy", "auto_merge"),
        merge_units_enabled=True,
        slay_relpath="SLAy_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_reports_enabled=True,
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=True,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "ok"
    assert summary.get("replot_only") is True
    assert summary.get("force_replot") is True
    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 1
    assert report_calls[0].get("after_count") == 1
    assert report_calls[0].get("applied_mappings_count") == 1
    assert "merge.report.unit_locations_before_after_png" in result.outputs


def test_run_spikesort_merge_stage_force_replot_only_falls_back_to_applied_operations(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    h5_path = tmp_path / "raw_data" / "input.raw.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"")

    well_out_dir = tmp_path / "well001"
    merge_out_dir = well_out_dir / "spikesort_outputs" / "SLAy_outputs"
    merge_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_json = merge_out_dir / "merge_metadata_summary.json"
    metadata_json.write_text(
        json.dumps(
            {
                "before": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "190": {"x_um": 2653.33, "y_um": 2079.95},
                            "195": {"x_um": 2653.45, "y_um": 2080.30},
                        }
                    }
                },
                "after": {
                    "analyzer": {
                        "unit_locations_by_unit": {
                            "105": {"x_um": 2653.69, "y_um": 2079.56},
                        }
                    }
                },
                "applied_merge_operations": [
                    {
                        "method": "slay",
                        "group_id": "202",
                        "pre_unit_ids": ["190", "195"],
                    }
                ],
                "applied_merge_group_count": 1,
                "change_validation": {"passes": True},
            }
        ),
        encoding="utf-8",
    )

    summary_json = merge_out_dir / "merge_stage_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "methods": [
                    {"name": "slay", "status": "ok"},
                ],
                "merge_metadata_summary_json": str(metadata_json),
                "outputs": {
                    "merge.metadata_summary_json": str(metadata_json),
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("merge methods must not run in force_replot-only mode")

    report_calls: list[dict[str, object]] = []

    def _fake_write_reports(*, merge_out_dir, before_snapshot, after_snapshot, applied_unit_mappings, stage_config):
        report_calls.append(
            {
                "before_count": len(before_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "after_count": len(after_snapshot.get("analyzer", {}).get("unit_locations_by_unit", {})),
                "applied_mappings_count": len(list(applied_unit_mappings or [])),
            }
        )
        return {
            "status": "ok",
            "before_unit_locations_count": 2,
            "after_unit_locations_count": 1,
            "outputs": {
                "merge.report.unit_locations_before_after_png": str(merge_out_dir / "unit_locations_before_after_merge.png"),
            },
        }

    monkeypatch.setattr(spikesort_runner, "compute_mea_analysis_output_dir", lambda **kwargs: well_out_dir)
    monkeypatch.setattr(spikesort_runner, "_run_slay_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_run_auto_merge_method", _fail_if_called)
    monkeypatch.setattr(spikesort_runner, "_write_merge_unit_location_reports", _fake_write_reports)

    stage_cfg = SimpleNamespace(
        merge_sequence=("SLAy",),
        merge_units_enabled=True,
        slay_relpath="SLAy_outputs",
        merge_metadata_enabled=True,
        merge_metadata_write_json=True,
        merge_metadata_json_relpath="merge_metadata_summary.json",
        merge_reports_enabled=True,
        merge_reports_2panel_enabled=True,
    )

    result = run_spikesort_merge_stage(
        h5_path=h5_path,
        stream_id="well001",
        mea_output_root=tmp_path,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        force_replot=True,
    )

    summary = _read_json(result.summary_json)

    assert summary.get("status") == "ok"
    assert summary.get("replot_only") is True
    assert len(report_calls) == 1
    assert report_calls[0].get("before_count") == 2
    assert report_calls[0].get("after_count") == 1
    assert report_calls[0].get("applied_mappings_count") == 1
    assert "merge.report.unit_locations_before_after_png" in result.outputs


def test_run_auto_merge_method_writes_per_iteration_outputs(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAnalyzer:
        def __init__(self, unit_ids: list[int]) -> None:
            self.unit_ids = list(unit_ids)

        def merge_units(self, *, merge_unit_groups, format, merging_mode, raise_error_if_overlap_fails):
            # Simulate one merge by dropping one unit id.
            if len(self.unit_ids) > 1:
                return _FakeAnalyzer(self.unit_ids[:-1])
            return _FakeAnalyzer(list(self.unit_ids))

        def save_as(self, *, format, folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "marker.txt").write_text("saved", encoding="utf-8")
            return self

    analyzer_state = {"calls": 0}

    def _fake_load_or_recompute(*, si_module, well_out_dir, stage_output_root_dir, sorter_output_dir, stage_config):
        return _FakeAnalyzer([1, 2, 3]), stage_output_root_dir / "analyzer_output", False

    def _fake_compute_groups(*, sorting_analyzer, template_diff_thresh):
        analyzer_state["calls"] += 1
        if analyzer_state["calls"] == 1:
            return [["1", "2"]]
        return []

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(spikesort_runner, "_load_or_recompute_spikesort_analyzer", _fake_load_or_recompute)
    monkeypatch.setattr(spikesort_runner, "_compute_auto_merge_groups", _fake_compute_groups)

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    stage_output_root_dir.mkdir(parents=True, exist_ok=True)

    stage_cfg = SimpleNamespace(
        auto_merge_enabled=True,
        auto_merge_relpath="automerge_outputs",
        auto_merge_delete_outputs_on_force_restart=True,
        auto_merge_candidate_pairs_reldir="recommended_merge_candidates",
        auto_merge_merged_units_reldir="merged_units",
        auto_merge_auto_accept_merges=True,
        auto_merge_template_diff_thresholds=(0.05,),
    )

    report = spikesort_runner._run_auto_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=stage_output_root_dir / "sorter_output",
    )

    assert report.get("status") == "ok"
    out_dir = Path(str(report.get("out_dir")))
    summary = _read_json(out_dir / "auto_merge_method_summary.json")

    assert summary.get("n_iterations") == 2
    assert summary.get("n_candidate_groups_total") == 1
    assert (out_dir / "recommended_merge_candidates" / "iteration_001.json").exists()
    assert (out_dir / "recommended_merge_candidates" / "iteration_001.tsv").exists()
    assert (out_dir / "recommended_merge_candidates" / "iteration_002.json").exists()
    assert (out_dir / "recommended_merge_candidates" / "iteration_002.tsv").exists()


def test_run_auto_merge_method_auto_accept_applies_at_each_threshold(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAnalyzer:
        def __init__(self, unit_ids: list[int], merge_calls: list[list[str]]) -> None:
            self.unit_ids = list(unit_ids)
            self._merge_calls = merge_calls

        def merge_units(self, *, merge_unit_groups, format, merging_mode, raise_error_if_overlap_fails):
            flat_groups = ["|".join(str(u) for u in group) for group in list(merge_unit_groups or [])]
            self._merge_calls.extend([flat_groups])
            # Return a new analyzer with one fewer unit to indicate progress.
            next_ids = self.unit_ids[:-1] if len(self.unit_ids) > 1 else list(self.unit_ids)
            return _FakeAnalyzer(next_ids, self._merge_calls)

        def save_as(self, *, format, folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "marker.txt").write_text("saved", encoding="utf-8")
            return self

    merge_calls: list[list[str]] = []
    calls_per_threshold: dict[float, int] = {}

    def _fake_load_or_recompute(*, si_module, well_out_dir, stage_output_root_dir, sorter_output_dir, stage_config):
        return _FakeAnalyzer([1, 2, 3, 4], merge_calls), stage_output_root_dir / "analyzer_output", False

    def _fake_compute_groups(*, sorting_analyzer, template_diff_thresh):
        key = float(template_diff_thresh)
        calls_per_threshold[key] = int(calls_per_threshold.get(key, 0)) + 1
        # First pass at each threshold proposes one merge; second pass terminates recursion.
        if int(calls_per_threshold[key]) == 1:
            if abs(key - 0.05) < 1e-9:
                return [["1", "2"]]
            if abs(key - 0.15) < 1e-9:
                return [["3", "4"]]
        return []

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(spikesort_runner, "_load_or_recompute_spikesort_analyzer", _fake_load_or_recompute)
    monkeypatch.setattr(spikesort_runner, "_compute_auto_merge_groups", _fake_compute_groups)

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    stage_output_root_dir.mkdir(parents=True, exist_ok=True)

    stage_cfg = SimpleNamespace(
        auto_merge_enabled=True,
        auto_merge_relpath="automerge_outputs",
        auto_merge_delete_outputs_on_force_restart=True,
        auto_merge_candidate_pairs_reldir="recommended_merge_candidates",
        auto_merge_merged_units_reldir="merged_units",
        auto_merge_auto_accept_merges=True,
        auto_merge_template_diff_thresholds=(0.05, 0.15),
    )

    report = spikesort_runner._run_auto_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=False,
        sorter_output_dir=stage_output_root_dir / "sorter_output",
    )

    assert report.get("status") == "ok"
    out_dir = Path(str(report.get("out_dir")))
    summary = _read_json(out_dir / "auto_merge_method_summary.json")

    assert summary.get("n_applied_groups_total") == 2
    assert summary.get("n_iterations") == 4
    assert len(merge_calls) == 2
    assert calls_per_threshold.get(0.05) == 2
    assert calls_per_threshold.get(0.15) == 2


def test_run_auto_merge_method_deletes_only_auto_merge_output_dir_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner

    class _FakeAnalyzer:
        def __init__(self, unit_ids: list[int]) -> None:
            self.unit_ids = list(unit_ids)

        def merge_units(self, *, merge_unit_groups, format, merging_mode, raise_error_if_overlap_fails):
            return _FakeAnalyzer(list(self.unit_ids))

        def save_as(self, *, format, folder):
            folder.mkdir(parents=True, exist_ok=True)
            return self

    def _fake_load_or_recompute(*, si_module, well_out_dir, stage_output_root_dir, sorter_output_dir, stage_config):
        return _FakeAnalyzer([1, 2]), stage_output_root_dir / "analyzer_output", False

    monkeypatch.setattr(spikesort_runner, "_import_spikeinterface_full_module", lambda: object())
    monkeypatch.setattr(spikesort_runner, "_load_or_recompute_spikesort_analyzer", _fake_load_or_recompute)
    monkeypatch.setattr(spikesort_runner, "_compute_auto_merge_groups", lambda **kwargs: [])

    well_out_dir = tmp_path / "well001"
    stage_output_root_dir = well_out_dir / "spikesort_outputs"
    stage_output_root_dir.mkdir(parents=True, exist_ok=True)

    auto_merge_out_dir = stage_output_root_dir / "automerge_outputs"
    auto_merge_out_dir.mkdir(parents=True, exist_ok=True)
    (auto_merge_out_dir / "stale.txt").write_text("old", encoding="utf-8")

    unrelated_dir = stage_output_root_dir / "SLAy_outputs"
    unrelated_dir.mkdir(parents=True, exist_ok=True)
    (unrelated_dir / "keep.txt").write_text("keep", encoding="utf-8")

    stage_cfg = SimpleNamespace(
        auto_merge_enabled=True,
        auto_merge_relpath="automerge_outputs",
        auto_merge_delete_outputs_on_force_restart=True,
        auto_merge_candidate_pairs_reldir="recommended_merge_candidates",
        auto_merge_merged_units_reldir="merged_units",
        auto_merge_auto_accept_merges=False,
        auto_merge_template_diff_thresholds=(0.05,),
    )

    report = spikesort_runner._run_auto_merge_method(
        well_out_dir=well_out_dir,
        stage_output_root_dir=stage_output_root_dir,
        output_rel_root="spikesort_outputs",
        stage_config=stage_cfg,
        force_restart=True,
        sorter_output_dir=stage_output_root_dir / "sorter_output",
    )

    out_dir = Path(str(report.get("out_dir")))
    summary = _read_json(out_dir / "auto_merge_method_summary.json")

    assert str(auto_merge_out_dir) in list(summary.get("removed_on_force_restart", []))
    assert str(auto_merge_out_dir) in list(report.get("removed_on_force_restart", []))
    assert not (auto_merge_out_dir / "stale.txt").exists()
    assert (unrelated_dir / "keep.txt").exists()
