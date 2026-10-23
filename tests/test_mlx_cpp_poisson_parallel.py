"""Independent default-optimized Brian CPP constant Poisson and checkpoints."""
from pathlib import Path
import subprocess
import numpy as np
import pytest


@pytest.fixture(scope='module')
def cpp_reference(tmp_path_factory,request):
    import brian2 as b
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.codegen.generators.cpp_generator import poisson_code
    directory=tmp_path_factory.mktemp('cpp-poisson-parallel-default')
    cache=b.prefs.codegen.runtime.cython.cache_dir
    b.prefs.codegen.runtime.cython.cache_dir=str(directory/'cython-default')
    request.addfinalizer(lambda:setattr(b.prefs.codegen.runtime.cython,'cache_dir',cache))
    text=(Path(b.__file__).parent/'devices/cpp_standalone/templates/objects.cpp').read_text()
    klass=text[text.index('class RandomGenerator {'):text.index('// In OpenMP')].replace('private:','public:',1)
    code='#include <random>\n#include <fstream>\n#include <iomanip>\n#include <cmath>\n#include <cstdlib>\nnamespace {\n'+klass
    code+='\nRandomGenerator rng;double _rand(int){return rng.rand();}\n'+poisson_code+r'''
    void dump(std::ostream& out) {
        auto copy=rng.gen;for(int j=0;j<624;++j)out<<copy()<<' ';
        out<<rng.has_stored_gauss<<' '<<std::hexfloat<<rng.stored_gauss<<'\n';
    }
    }
    int main(int argc,char**argv) {
        double lam=strtod(argv[1],nullptr);int count=atoi(argv[2]),width=atoi(argv[3]);
        rng.seed(81);rng.randn();std::ofstream values(argv[4],std::ios::binary),states(argv[5]);
        dump(states);
        for(int j=0;j<count;++j) {int32_t value=_poisson(lam,0);values.write((char*)&value,4);
            int end=j+1;if((width&&end%width==0)||(!width&&(end==1||end==2||end==7||end==257))||end==count)dump(states);}
    }
    '''
    (directory/'probe.cpp').write_text(code);_,flags=get_compiler_and_args()
    subprocess.run(['c++',*flags,str(directory/'probe.cpp'),'-o',str(directory/'probe')],check=True,capture_output=True)
    references={}
    def get(lam,count,width=0):
        key=(lam,count,width)
        if key not in references:
            prefix=directory/f'{lam}-{count}-{width}'
            subprocess.run([str(directory/'probe'),str(lam),str(count),str(width),str(prefix)+'.bin',str(prefix)+'.txt'],check=True,capture_output=True)
            states=[]
            for line in Path(str(prefix)+'.txt').read_text().splitlines():
                row=line.split();assert len(row)==626
                states.append((np.asarray(row[:624],np.uint32),int(row[624]),float.fromhex(row[625])))
            references[key]=np.fromfile(str(prefix)+'.bin',np.int32),states
        return references[key]
    return get


def initial():
    from mlx_backend.expression_rng import seeded,normals
    _,state=normals(seeded([81]),1,target='cpp')
    return state


def assert_state(actual,expected):
    from mlx_backend.expression_rng import to_numpy_states
    state=to_numpy_states(actual)[0];copy=np.random.RandomState();copy.set_state(state)
    np.testing.assert_array_equal(copy.randint(0,2**32,624,dtype=np.uint32),expected[0]);assert state[3:]==expected[1:]


@pytest.mark.metal
@pytest.mark.parametrize('lam',[-1.,0.,1.,3.,17.,100.,1e16])
def test_cpp_parallel_constant_poisson_and_native_exhaustion(lam,cpp_reference):
    import mlx.core as mx
    from mlx_backend.expression_poisson_parallel import constant_poisson
    from mlx_backend.expression_rng import MT19937State
    count=1000003;expected,states=cpp_reference(lam,count)
    values,end,errors=constant_poisson(initial(),count,lam,target='cpp')
    assert values.dtype==mx.int32;np.testing.assert_array_equal(np.asarray(errors),0)
    np.testing.assert_array_equal(np.asarray(values).reshape(-1),expected);assert_state(end,states[-1])
    # The same independent CPU prefix checks masked counts and forced native
    # exhaustion without generating another large reference.
    start=initial();repeat=lambda a:mx.broadcast_to(a.reshape(1,-1),(5,a.size)).reshape(-1)
    start=MT19937State(repeat(start.words),repeat(start.position),5,repeat(start.has_gauss),repeat(start.gauss))
    counts=(0,1,2,7,257)
    values,end,errors=constant_poisson(start,264,lam,mx.array(counts,mx.uint32),raw_capacity=1,target='cpp')
    actual=np.asarray(values);np.testing.assert_array_equal(np.asarray(errors),0)
    from mlx_backend.expression_rng import from_numpy_states,to_numpy_states
    for row,(n,snapshot,want)in enumerate(zip(counts,to_numpy_states(end),states[:5])):
        np.testing.assert_array_equal(actual[row,:n],expected[:n]);np.testing.assert_array_equal(actual[row,n:],0)
        assert_state(from_numpy_states([snapshot]),want)


@pytest.mark.metal
@pytest.mark.parametrize('lam',[-1.,0.,1.,3.,17.,100.,1e16])
def test_cpp_poisson_pool_every_frame_checkpoint(lam,cpp_reference):
    from mlx_backend.expression_poisson_parallel import poisson_pool
    width,frames=1009,37;expected,states=cpp_reference(lam,width*frames,width)
    pool=poisson_pool(initial(),width,frames,lam,target='cpp')
    np.testing.assert_array_equal(np.asarray(pool.values).reshape(-1),expected)
    for frame,want in enumerate(states[1:]):assert_state(pool.state_at(frame),want)
