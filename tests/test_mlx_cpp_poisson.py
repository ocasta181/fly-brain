"""Brian CPP Poisson source, default CPU optimizations, exact native streams."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from mlx_backend.expression_rng import seeded,normals,to_numpy_states


@pytest.mark.metal
def test_cpp_poisson_values_int32_edges_and_full_stream_match_cpu(tmp_path):
    import brian2 as b
    import mlx.core as mx
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.codegen.generators.cpp_generator import poisson_code
    from mlx_backend.expression_poisson_compiled import poisson
    text=(Path(b.__file__).parent/'devices/cpp_standalone/templates/objects.cpp').read_text()
    klass=text[text.index('class RandomGenerator {'):text.index('// In OpenMP')].replace('private:','public:',1)
    code='#include <random>\n#include <fstream>\n#include <iomanip>\n#include <cmath>\n#include <cstdlib>\n#include <string>\n'+klass
    code+='\nRandomGenerator rng;double _rand(int){return rng.rand();}\n'+poisson_code+r'''
    int main(int argc,char**argv) {
        rng.seed(81);rng.randn();std::ifstream in(argv[1]);
        std::ofstream values(argv[2],std::ios::binary),states(argv[3]);std::string line;
        while(std::getline(in,line)) {
            int32_t value=_poisson(strtod(line.c_str(),nullptr),0);values.write((char*)&value,4);
            auto copy=rng.gen;for(int j=0;j<624;++j)states<<copy()<<' ';
            states<<rng.has_stored_gauss<<' '<<std::hexfloat<<rng.stored_gauss<<'\n';
        }
    }
    '''
    (tmp_path/'probe.cpp').write_text(code);_,flags=get_compiler_and_args()
    subprocess.run(['c++',*flags,str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')],check=True,capture_output=True)
    rates=np.tile(np.array([0.,-0.,1e-12,.01,.3,1.,3.,np.nextafter(10.,0),10.,np.nextafter(10.,np.inf),
        17.,100.,1000.,1e8,1e16,-1.,-1e10,np.nan,np.inf]),17)
    (tmp_path/'rates.txt').write_text(''.join(str(x)+'\n'for x in rates))
    subprocess.run([str(tmp_path/'probe'),str(tmp_path/'rates.txt'),str(tmp_path/'values.bin'),
        str(tmp_path/'states.txt')],check=True,capture_output=True)
    expected=np.fromfile(tmp_path/'values.bin',np.int32)
    snapshots=[line.split()for line in(tmp_path/'states.txt').read_text().splitlines()]
    _,state=normals(seeded([81]),1,target='cpp')
    for begin in range(0,len(rates),19):
        actual,state=poisson(state,mx.array(rates[None,begin:begin+19].view(np.uint64)))
        assert actual.dtype==mx.int32
        np.testing.assert_array_equal(np.asarray(actual).reshape(-1),expected[begin:begin+19])
        got=to_numpy_states(state)[0];copy=np.random.RandomState();copy.set_state(got)
        want=snapshots[begin+18];assert len(want)==626
        np.testing.assert_array_equal(copy.randint(0,2**32,624,dtype=np.uint32),np.asarray(want[:624],np.uint32))
        assert got[3:]==(int(want[624]),float.fromhex(want[625]))
