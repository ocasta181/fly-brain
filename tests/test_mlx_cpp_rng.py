"""Independent frozen Brian C++ RNG class, compiled with default CPU flags."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from mlx_backend.expression_rng import seeded,normals,to_numpy_states


@pytest.fixture(scope='module')
def cpp_normal_reference(tmp_path_factory):
    import brian2 as b
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    directory=tmp_path_factory.mktemp('cpp-rng-default')
    source=(Path(b.__file__).parent/'devices/cpp_standalone/templates/objects.cpp').read_text()
    klass=source[source.index('class RandomGenerator {'):source.index('// In OpenMP')]
    # Add a reader only; the class's seed/draw implementation stays untouched.
    klass=klass.replace('public:',r'''public:
    void dump(std::ostream& out,bool initialized) {
        auto future=gen;out<<std::dec<<std::defaultfloat;
        for(int j=0;j<624;++j)out<<future()<<' ';
        out<<has_stored_gauss<<' '<<std::hexfloat<<(initialized?stored_gauss:0.0)<<'\n';
    }
    ''',1)
    code='#include <random>\n#include <fstream>\n#include <cmath>\n#include <iomanip>\n'+klass+r'''
    int main(int argc,char**argv) {
        RandomGenerator rng;rng.seed(81);
        std::ofstream values(argv[1],std::ios::binary),states(argv[2]);
        int chunks[]={0,1,2,7,512,3575};int total=0;
        for(int count:chunks) {
            for(int j=0;j<count;++j) {double v=rng.randn();values.write((char*)&v,sizeof(v));}
            total+=count;rng.dump(states,total>0);
        }
        rng.seed(81);std::ofstream frames(argv[3]);
        for(int frame=0;frame<241;++frame) {for(int j=0;j<17;++j)rng.randn();rng.dump(frames,true);}
    }
    '''
    (directory/'probe.cpp').write_text(code)
    _,flags=get_compiler_and_args()
    subprocess.run(['c++',*flags,str(directory/'probe.cpp'),'-o',str(directory/'probe')],check=True,capture_output=True)
    subprocess.run([str(directory/'probe'),str(directory/'values.bin'),str(directory/'states.txt'),
        str(directory/'frames.txt')],check=True,capture_output=True)
    def read(path):
        result=[]
        for line in path.read_text().splitlines():
            items=line.split();assert len(items)==626
            result.append((np.asarray(items[:624],np.uint32),int(items[624]),float.fromhex(items[625])))
        return result
    return np.fromfile(directory/'values.bin',np.float64),read(directory/'states.txt'),read(directory/'frames.txt')


def assert_state(actual,expected):
    got=to_numpy_states(actual)[0]
    # libc++ stores a rotated incremental MT state; NumPy twists complete
    # blocks. All624future engine words verify the full equivalent stream.
    copy=np.random.RandomState();copy.set_state(got)
    np.testing.assert_array_equal(copy.randint(0,2**32,624,dtype=np.uint32),expected[0])
    assert got[3:]==expected[1:]


@pytest.mark.metal
def test_cpp_normal_serial_values_full_state_and_cache_match_cpu(cpp_normal_reference):
    values,states,_=cpp_normal_reference;state=seeded([81]);begin=0
    for count,expected in zip((0,1,2,7,512,3575),states):
        actual,state=normals(state,count,target='cpp')
        np.testing.assert_array_equal(np.asarray(actual).reshape(-1).view(np.float64),values[begin:begin+count])
        assert_state(state,expected);begin+=count
    assert np.count_nonzero(values!=np.random.RandomState(81).randn(len(values)))>0


@pytest.mark.metal
@pytest.mark.parametrize('candidate_count',[None,1])
def test_cpp_parallel_values_and_each_pool_checkpoint_match_cpu(cpp_normal_reference,candidate_count):
    from mlx_backend.expression_rng_parallel import parallel_normals,normal_pool
    values,states,frames=cpp_normal_reference
    actual,state=parallel_normals(seeded([81]),len(values),candidate_count=candidate_count,target='cpp')
    np.testing.assert_array_equal(np.asarray(actual).reshape(-1).view(np.float64),values)
    assert_state(state,states[-1])
    pool=normal_pool(seeded([81]),17,241,candidate_count=candidate_count,target='cpp')
    np.testing.assert_array_equal(np.asarray(pool.values).reshape(-1).view(np.float64),values)
    for frame,expected in enumerate(frames):assert_state(pool.state_at(frame),expected)
