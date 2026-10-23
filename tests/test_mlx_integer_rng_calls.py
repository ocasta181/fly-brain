"""Original Cython integer-valued calls inside stateful expression operands."""
import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('conversion',['draws','boundaries','contraction','nested','timed','constant'])
def test_original_cython_integer_calls_and_continuation(tmp_path,monkeypatch,conversion):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    def build(mode):
        integer_prefix=('rem=(int(7*rand())+a)%den;quo=(int(9*randn())-a)//den;'
            if conversion=='draws'else'rem=int(x)%den;quo=(int(x)+a)//den;')
        if conversion=='contraction':
            integer_prefix='rem=int(2*(x*y+z))%den;quo=(int(2*(x*y+z))+a)//den;'
        elif conversion=='nested':
            integer_prefix='rem=int(poisson(2)+rand())%den;quo=(int(9*randn())-a)//den;'
        elif conversion=='timed':
            integer_prefix='rem=int(drive(t+i*dt)+rand())%den;quo=(int(9*randn())-a)//den;'
        elif conversion=='constant':
            integer_prefix='rem=int(9*randn())%(-3);quo=a//(-3);'
        initial={'a':[-7,7,-4,4],'den':[-3,2,3,-2],
            'x':[-np.inf,-2147483649.25,2147483648.5,np.nan],'y':0.,'z':0.}
        if conversion=='contraction':
            initial.update(x=[1+2**-27,1-2**-27,-1-2**-27,-1+2**-27],
                y=[1-2**-27,1+2**-27,1-2**-27,1+2**-27],z=[-.5,-.5,.5,.5])
        parameters=dict(
            eqs='v:volt\ng:volt\nrfc:second\na:integer\nden:integer\nrem:integer\nquo:integer\nx:1\ny:1\nz:1',
            eq_th='rand()<1',
            eq_rst=integer_prefix+'a=rem+poisson(2);v=v_rst+(quo+randn())*mV;g=0*mV',
            t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms)
        if conversion=='timed':
            parameters['drive']=b.TimedArray([-.75,3.5,-3.5,2**31+.5],dt=.2*b.ms)
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=parameters,
            silenced=(),initial_state=initial)
        return fixture

    def exercise(fixture, mode):
        for duration in (.2,0.,.3):
            yield observe_run(fixture,duration,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=829,target='cython')


@pytest.mark.metal
@pytest.mark.parametrize('prefix_draw',[False,True])
def test_original_cython_checked_integer_call_repair(tmp_path,monkeypatch,prefix_draw):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    def build(mode):
        expression=('randn()+'if prefix_draw else'')+'1/den+rand()'
        fixture=build_original_api_fixture(tmp_path/str(mode=='native'),parameters=dict(
            eqs='v:volt\ng:volt\nrfc:second\nden:integer\nrem:integer',
            eq_th='rand()<1',eq_rst=f'rem=int({expression});v=v_rst+randn()*mV;g=0*mV',
            t_rfc=0*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz,t_dly=.2*b.ms),
            silenced=(),initial_state={'den':[-3,2,3,-2]})
        return fixture

    def exercise(fixture, mode):
        yield observe_run(fixture,.2,rng=True)
        fixture.neurons.den[1]=0
        yield observe_run(fixture,.3,raises=b.BrianObjectException,rng=True)
        fixture.neurons.den=[-3,2,3,-2]
        yield observe_run(fixture,.3,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=829,target='cython')
