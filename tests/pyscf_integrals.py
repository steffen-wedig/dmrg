from dmrg.hf_integrals import get_H2_integrals



h1e, h2e = get_H2_integrals()

print(h2e[0,0,1,1])

print(h2e.reshape((16,16)))