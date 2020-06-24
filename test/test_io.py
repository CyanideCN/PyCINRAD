from io import BytesIO

import pytest

from cinrad.io.level2 import infer_type, CinradReader
from cinrad.error import RadarDecodeError

def test_infer_type_from_fname():
    fill = bytes(200)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, 'Z_RADR_I_Z9200_20000000000000_O_DOR_SA_CAP.bin')
    fake_file.close()
    assert code == 'Z9200'
    assert _type == 'SA'

def test_infer_type_from_file_sc():
    fill = bytes(100) + b"CINRAD/SC" + bytes(50)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, "foo")
    fake_file.close()
    assert code == None
    assert _type == 'SC'

def test_infer_type_from_file_cd():
    fill = bytes(100) + b"CINRAD/CD" + bytes(50)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, "foo")
    fake_file.close()
    assert code == None
    assert _type == 'CD'

def test_infer_type_from_file_cc():
    fill = bytes(116) + b"CINRAD/CC" + bytes(50)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, "foo")
    fake_file.close()
    assert code == None
    assert _type == 'CC'

def test_infer_type_from_incomplete_fname():
    fill = bytes(200)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, 'Z_RADR_I_Z9200.bin')
    fake_file.close()
    assert code == None
    assert _type == None

def test_missing_radar_type():
    fill = bytes(200)
    fake_file = BytesIO(fill)
    with pytest.raises(RadarDecodeError):
        CinradReader(fake_file)