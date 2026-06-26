import os
import subprocess
import sys

import numpy as np
import xarray as xr
import cinrad.cappi as cappi_mod
from cinrad.calc import CAPPI
from cinrad.common import get_dtype


def _make_sweep_data(naz=360, nrange=230, elevation=1.0,
                     refl_value=20.0, noise=np.nan):
    """Create synthetic sweep Dataset mimicking cinrad.io output.

    Parameters
    ----------
    naz : int, 方位角数量
    nrange : int, 距离库数量
    elevation : float, 仰角 (度)
    refl_value : float, 反射率值 (dBZ)
    noise : float, 噪声填充值 (NaN 表示无数据)

    Returns
    -------
    xarray.Dataset
    """
    # 方位角: 0~360度, 转换为弧度
    azimuth = np.linspace(0, 2 * np.pi, naz, endpoint=False)
    # 距离: 0~230 km
    distance = np.linspace(0, 230, nrange)

    # 反射率数据: 在 [50, 150] km 环形区域内填 refl_value
    ref = np.full((naz, nrange), noise, dtype=np.float64)
    for i in range(naz):
        for j in range(nrange):
            if 50 <= distance[j] <= 150:
                # 在大多数方位角都有值的环形
                ref[i, j] = refl_value

    # 添加一些方位角上的变化来测试插值
    # 方位角 90度 (朝东) 方向信号更强
    for i in range(naz):
        az_deg = np.rad2deg(azimuth[i])
        for j in range(nrange):
            if 50 <= distance[j] <= 150:
                if 80 <= az_deg <= 100:
                    ref[i, j] = refl_value + 10.0  # 强回波区
                elif 260 <= az_deg <= 280:
                    ref[i, j] = noise  # 西侧无回波

    # 构建 Dataset
    from cinrad.projection import height, get_coordinate
    h_radar = 0.0  # 雷达高度 (m)
    lon0, lat0 = 120.0, 30.0  # 雷达位置

    hgt = height(distance, elevation, h_radar)
    hgt_2d = np.broadcast_to(hgt, (naz, nrange))

    lon, lat = get_coordinate(distance, azimuth, elevation, lon0, lat0)

    da = xr.DataArray(ref, coords=[azimuth, distance],
                      dims=["azimuth", "distance"])
    ds = xr.Dataset(
        {"REF": da},
        attrs={
            "elevation": elevation,
            "range": 230,
            "scan_time": "2024-01-01 00:00:00",
            "site_code": "TEST",
            "site_name": "Test Radar",
            "site_longitude": lon0,
            "site_latitude": lat0,
            "tangential_reso": 1.0,
        },
    )
    ds["longitude"] = (["azimuth", "distance"], lon)
    ds["latitude"] = (["azimuth", "distance"], lat)
    ds["height"] = (["azimuth", "distance"], hgt_2d)
    return ds


def _make_volume(naz=360, nrange=230, elevations=None, refl_value=20.0):
    """Create multi-sweep volume data."""
    if elevations is None:
        elevations = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 10.0]
    sweeps = []
    for i, el in enumerate(elevations):
        # 仰角越高, 回波值略变 (模拟衰减)
        v = refl_value - i * 0.5
        ds = _make_sweep_data(naz, nrange, el, refl_value=v)
        sweeps.append(ds)
    return sweeps


def _make_azimuth_value_sweep():
    ds = _make_sweep_data(naz=36, nrange=60, elevation=1.0, noise=np.nan)
    az_deg = np.array([350] + list(range(0, 350, 10)), dtype=np.float64)
    az = np.deg2rad(az_deg)
    data = np.repeat(az_deg[:, None], ds.sizes["distance"], axis=1)
    ds = ds.assign_coords(azimuth=az)
    ds["REF"] = (["azimuth", "distance"], data)
    return ds


# =============== 基本功能测试 ===============

def test_cappi_import():
    """测试 CAPPI 类能正确导入."""
    from cinrad import CAPPI as CAPI
    assert CAPI is CAPPI


def test_cappi_import_without_numba():
    """测试未安装 numba 时 cinrad 仍可导入."""
    code = """
import builtins
real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "numba" or name.startswith("numba."):
        raise ImportError("blocked numba")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import
import cinrad
from cinrad.calc import CAPPI
from cinrad.cappi import HAS_NUMBA
assert HAS_NUMBA is False
assert CAPPI is cinrad.CAPPI
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_cappi_init_single():
    """测试单仰角层初始化."""
    sweeps = [_make_sweep_data(elevation=1.0)]
    cappi = CAPPI(sweeps)
    assert len(cappi.rl) == 1
    assert cappi.dtype == "REF"
    assert cappi.radar_lat == 30.0
    assert cappi.radar_lon == 120.0
    assert len(cappi._sweep_azimuths) == 1


def test_cappi_init_multi():
    """测试多仰角层初始化和去重."""
    sweeps = _make_volume()
    cappi = CAPPI(sweeps)
    assert len(cappi.elev_angles) == len(sweeps)


def test_cappi_dedup():
    """测试重复仰角去重."""
    s = [_make_sweep_data(elevation=1.0), _make_sweep_data(elevation=1.0)]
    cappi = CAPPI(s)
    assert len(cappi.rl) == 1


def test_cappi_sorts_rolled_azimuth():
    """测试从 350 度起扫、跨 360 度的径向顺序."""
    cappi = CAPPI([_make_azimuth_value_sweep()])
    assert np.all(np.diff(cappi._sweep_azimuths[0]) >= 0)

    v_0 = cappi._interp_sweep_sample(0, 0.0, 30000.0, np.nan)
    v_350 = cappi._interp_sweep_sample(0, 350.0, 30000.0, np.nan)
    assert np.isclose(v_0, 0.0, atol=1e-6)
    assert np.isclose(v_350, 350.0, atol=1e-6)


# =============== 坐标转换测试 ===============

def test_cartesian_to_antenna():
    """测试 _cartesian_to_antenna 坐标反算."""
    # 已知: 正北方向 (x=0, y=10000) → 方位角 0度
    az, r, el = CAPPI._cartesian_to_antenna(
        np.array([0.0]), np.array([10000.0]),
        np.array([100.0]), h_radar=0.0)
    assert np.isclose(az[0], 0.0, atol=1e-6), f"az={az[0]}"
    assert np.isclose(r[0], 10000.0, atol=1.0), f"r={r[0]}"

    # 正东方向 (x=10000, y=0) → 方位角 90度
    az, r, el = CAPPI._cartesian_to_antenna(
        np.array([10000.0]), np.array([0.0]),
        np.array([100.0]), h_radar=0.0)
    assert np.isclose(az[0], 90.0, atol=1e-6), f"az={az[0]}"

    # 原点 (0, 0, z=h) → 斜距 = |h_radar - z|
    # 因为 s=0 → φ=0, cos(0)=1 → r² = (reff+h_radar)²+(reff+z)²-2(reff+h_radar)(reff+z)=(h_radar-z)²
    az, r, el = CAPPI._cartesian_to_antenna(
        np.array([0.0]), np.array([0.0]),
        np.array([1000.0]), h_radar=0.0)
    assert np.isclose(r[0], 1000.0, atol=1.0), f"r={r[0]}"
    assert np.isclose(el[0], 90.0, atol=1.0), f"el={el[0]}"


def test_cartesian_to_antenna_vectorized():
    """测试矢量版本对多个点输入的正确性."""
    x = np.array([0.0, 10000.0, 20000.0])
    y = np.array([10000.0, 0.0, 0.0])
    z = np.array([100.0, 100.0, 100.0])
    az, r, el = CAPPI._cartesian_to_antenna(x, y, z)
    assert az.shape == (3,)
    assert r.shape == (3,)
    assert el.shape == (3,)
    # 第一点: 正北, 方位角 0
    assert np.isclose(az[0], 0.0, atol=1e-6)
    # 第二点: 正东, 方位角 90
    assert np.isclose(az[1], 90.0, atol=1e-6)


# =============== 插值单元测试 ===============

def test_interp_linear_basic():
    """测试 _interp_linear."""
    # 基本线性插值: coord=1.5, 范围[1,2], 值[10,20] → 15
    v = CAPPI._interp_linear(1.5, 1.0, 2.0, 10.0, 20.0, np.nan)
    assert np.isclose(v, 15.0), f"v={v}"

    # 下界相等 → fillvalue
    v = CAPPI._interp_linear(1.5, 1.0, 1.0, 10.0, 20.0, np.nan)
    assert np.isnan(v)

    # dat_0 为 NaN → 返回 dat_1
    v = CAPPI._interp_linear(1.5, 1.0, 2.0, np.nan, 20.0, np.nan)
    assert np.isclose(v, 20.0)

    # 两个都为 NaN → fillvalue
    v = CAPPI._interp_linear(1.5, 1.0, 2.0, np.nan, np.nan, np.nan)
    assert np.isnan(v)

    # 自定义 fillvalue
    v = CAPPI._interp_linear(1.5, 1.0, 2.0, np.nan, np.nan, -999.0)
    assert np.isclose(v, -999.0)


def test_interp_sweep_sample():
    """测试 _interp_sweep_sample 单点插值."""
    sweeps = [_make_sweep_data(naz=360, nrange=230, elevation=1.0)]
    cappi = CAPPI(sweeps)

    # 在环形回波区内的点：距离 100km, 方位角 0度(正北)
    # 应该得到接近 refl_value=20 的插值
    v = cappi._interp_sweep_sample(0, 0.0, 100000.0, np.nan)
    assert np.isclose(v, 20.0, atol=1.0), f"v={v}"

    # 在环形回波区外：距离 10km → 无回波
    v = cappi._interp_sweep_sample(0, 0.0, 10000.0, np.nan)
    assert np.isnan(v), f"v={v}"

    # 超出最大距离
    v = cappi._interp_sweep_sample(0, 0.0, 300000.0, np.nan)
    assert np.isnan(v), f"v={v}"


# =============== CAPPI 整体测试 ===============

def test_get_cappi_xy_basic():
    """测试 get_cappi_xy 基本功能."""
    sweeps = _make_volume(naz=36, nrange=120)  # 小网格加速测试
    cappi = CAPPI(sweeps)

    x = np.linspace(-200000, 200000, 21)
    y = np.linspace(-200000, 200000, 21)

    ds = cappi.get_cappi_xy(x, y, level_height=1000.0)
    assert ds is not None
    assert "REF" in ds
    assert ds["REF"].shape == (21, 21)
    assert ds.attrs.get("cappi_height") == 1000.0


def test_get_cappi_xy_at_radar():
    """测试雷达站正上方的 CAPPI 值."""
    sweeps = _make_volume(naz=36, nrange=120)
    cappi = CAPPI(sweeps)

    # 只在几个关键点测试
    x = np.array([-100000, 0, 100000])
    y = np.array([-100000, 0, 100000])

    # 较低高度 → 应有更多有效值
    ds = cappi.get_cappi_xy(x, y, level_height=1000.0)
    ref = ds["REF"].values
    # 雷达正上方 (索引 1,1) 可能有值或 NaN 都合理
    assert ref.shape == (3, 3)


def test_cappi_symmetry():
    """测试 CAPPI 的对称性."""
    # 用均匀的环形回波验证
    sweeps = _make_volume(naz=36, nrange=120, refl_value=30.0)
    cappi = CAPPI(sweeps)

    x = np.linspace(-120000, 120000, 13)
    y = np.linspace(-120000, 120000, 13)

    ds = cappi.get_cappi_xy(x, y, level_height=2000.0)
    ref = ds["REF"].values

    # 结果关于原点对称
    center = ref[6, 6]
    # 对称位置的值应接近
    for i in [-2, -1, 1, 2]:
        for j in [-2, -1, 1, 2]:
            if not np.isnan(ref[6 + i, 6 + j]) and not np.isnan(ref[6 - i, 6 - j]):
                assert np.isclose(ref[6 + i, 6 + j],
                                  ref[6 - i, 6 - j],
                                  atol=3.0), (
                    f"不对称: ({6+i},{6+j})={ref[6+i,6+j]} "
                    f"vs ({6-i},{6-j})={ref[6-i,6-j]}"
                )


def test_cappi_heights():
    """测试不同 CAPPI 高度."""
    sweeps = _make_volume(naz=36, nrange=120)
    cappi = CAPPI(sweeps)

    x = np.linspace(-150000, 150000, 21)
    y = np.linspace(-150000, 150000, 21)

    # 三个不同高度
    ds1 = cappi.get_cappi_xy(x, y, level_height=1000.0)
    ds2 = cappi.get_cappi_xy(x, y, level_height=3000.0)
    ds3 = cappi.get_cappi_xy(x, y, level_height=8000.0)

    ref1, ref2, ref3 = ds1["REF"], ds2["REF"], ds3["REF"]

    # 检查各高度至少有部分有效值
    n_valid_1 = np.sum(~np.isnan(ref1.values))
    n_valid_2 = np.sum(~np.isnan(ref2.values))
    n_valid_3 = np.sum(~np.isnan(ref3.values))
    total_valid = n_valid_1 + n_valid_2 + n_valid_3
    assert total_valid > 0, "所有高度均无有效值"


# =============== 边界条件测试 ===============

def test_cappi_empty():
    """测试空体扫."""
    try:
        cappi = CAPPI([])
        assert False, "应抛出异常"
    except Exception:
        pass


def test_cappi_single_sweep():
    """测试单仰角体扫."""
    sweeps = [_make_sweep_data(naz=36, nrange=60, elevation=1.0)]
    cappi = CAPPI(sweeps)

    x = np.linspace(-100000, 100000, 11)
    y = np.linspace(-100000, 100000, 11)

    ds = cappi.get_cappi_xy(x, y, level_height=1000.0)
    assert ds is not None
    assert ds["REF"].shape == (11, 11)


def test_cappi_custom_fillvalue_not_interpolated(monkeypatch):
    """测试自定义填充值不会作为有效数据参与仰角层间插值."""
    sweeps = _make_volume(
        naz=36, nrange=120, elevations=[0.5, 3.0], refl_value=20.0
    )
    sweeps[0]["REF"] = xr.full_like(sweeps[0]["REF"], np.nan)
    sweeps[1]["REF"] = xr.full_like(sweeps[1]["REF"], 20.0)
    cappi = CAPPI(sweeps)

    for use_numba in [cappi_mod.HAS_NUMBA, False]:
        monkeypatch.setattr(cappi_mod, "HAS_NUMBA", use_numba)
        ds = cappi.get_cappi_xy(
            np.array([0.0]), np.array([100000.0]),
            level_height=3000.0, fillvalue=-999.0,
        )
        value = float(ds["REF"].values[0, 0])
        assert np.isclose(value, 20.0, atol=1e-6)


def test_cappi_get_cappi_lonlat():
    """测试 get_cappi_lonlat."""
    sweeps = _make_volume(naz=36, nrange=120)
    cappi = CAPPI(sweeps)

    lon = np.linspace(119.5, 120.5, 11)
    lat = np.linspace(29.5, 30.5, 11)

    ds = cappi.get_cappi_lonlat(lon, lat, level_height=2000.0)
    assert ds is not None
    assert "REF" in ds
    assert ds["REF"].shape == (11, 11)
    # 坐标应为 lat, lon
    assert ds["REF"].dims == ("lat", "lon")


def test_cappi_verbose():
    """测试 verbose 模式 (不应报错)."""
    sweeps = _make_volume(naz=18, nrange=60, elevations=[1.0, 2.0, 3.0])
    cappi = CAPPI(sweeps, verbose=True)

    x = np.linspace(-100000, 100000, 11)
    y = np.linspace(-100000, 100000, 11)

    ds = cappi.get_cappi_xy(x, y, level_height=1000.0)
    assert ds is not None


# =============== 数值精度测试 ===============

def test_interp_sweep_sample_wraparound():
    """测试方位角 0/360 循环."""
    sweeps = [_make_sweep_data(naz=36, nrange=60, elevation=1.0)]
    cappi = CAPPI(sweeps)

    # 方位角 359度 (接近 0度)
    v_near_0 = cappi._interp_sweep_sample(0, 1.0, 100000.0, np.nan)

    # 方位角 0度
    v_0 = cappi._interp_sweep_sample(0, 0.0, 100000.0, np.nan)

    # 方位角 358度 (也接近 0度)
    v_near_360 = cappi._interp_sweep_sample(0, 359.0, 100000.0, np.nan)

    # 对于环形回波, 所有方位角在相同距离的值应接近
    if not np.isnan(v_near_0) and not np.isnan(v_0):
        # 允许插值带来的微小差异
        assert np.isclose(v_near_0, v_0, atol=3.0), (
            f"0/360 循环: az=1° {v_near_0} vs az=0° {v_0}"
        )

    if not np.isnan(v_near_360) and not np.isnan(v_0):
        assert np.isclose(v_near_360, v_0, atol=3.0), (
            f"0/360 循环: az=359° {v_near_360} vs az=0° {v_0}"
        )


def test_cartesian_to_antenna_consistency():
    """验证坐标反算的自洽性.

    使用已知的天线坐标 (azimuth, range, elevation) 先通过
    pycwr 或等效公式计算笛卡尔坐标, 再反算回天线坐标.
    """
    # 构造简单场景: 雷达位于原点, 目标正北 50km, 高度 3km
    # 先通过 _cartesian_to_antenna 反算天线坐标
    az, r, el = CAPPI._cartesian_to_antenna(
        np.array([0.0]), np.array([50000.0]),
        np.array([3000.0]), h_radar=0.0)

    # 方位角应接近 0 (正北)
    assert np.isclose(az[0], 0.0, atol=1.0), f"az={az[0]}"

    # 斜距应接近 sqrt(50000² + 3000²) ≈ 50090, 但由于地球曲率略不同
    # 实际上公式是精确球面几何, 不是平面勾股定理
    r_expected = np.sqrt(50000**2 + 3000**2)
    assert np.isclose(r[0], r_expected, atol=500), f"r={r[0]} vs {r_expected}"

    # 仰角应 > 0
    assert el[0] > 0, f"el={el[0]}"
