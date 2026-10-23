"""Original Cython64 dynamic-power naming failures, after frontend composition."""
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('record',[False,True])
def test_original_cython_loop_local_power_with_aliases_and_poisson(tmp_path,monkeypatch,record):
    import test_mlx_identifiers as names
    configuration=names.identifier_configuration
    def original_power_configuration():
        result=configuration()
        result['eq_rst']=result['eq_rst'].replace('power_value=k+negative','power_value=k**negative')
        return result
    monkeypatch.setattr(names,'identifier_configuration',original_power_configuration)
    names.test_actual_identifier_aliases_poisson_recording_and_continuation(
        tmp_path,'cython','float64',record)
