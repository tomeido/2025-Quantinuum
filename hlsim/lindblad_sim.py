"""Utils for Lindbladian simulation."""

import numpy as np
from pytket import Qubit
from pytket.circuit import CircBox, Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator

from hlsim.hamiltonian_sim import (
    get_hamiltonian_simulation_circbox,
    get_xxz_chain_hamiltonian,
)
from hlsim.utils import get_pauli_exp_box_from_QubitPauliString


def get_second_order_dilation_trotter_step_from_QubitPauliOperator(
    qubit_pauli_operator: QubitPauliOperator, Delta_t: float, qubits: list[Qubit]
):
    circ = Circuit(name="Dilation Trotter step")
    reg_trotterbox = circ.add_q_register("reg_trotterbox", len(qubits))

    qubit_pauli_strings = list(qubit_pauli_operator._dict.keys())
    coefficients = list(qubit_pauli_operator._dict.values())

    pauli_exp_boxes = []
    for qps, coef in zip(qubit_pauli_strings[:-1], coefficients[:-1]):
        pauli_exp_boxes.append(
            get_pauli_exp_box_from_QubitPauliString(
                coef / 2, np.sqrt(Delta_t), qps, qubits
            )
        )

    pauli_exp_boxes.append(
        get_pauli_exp_box_from_QubitPauliString(
            coefficients[-1], np.sqrt(Delta_t), qubit_pauli_strings[-1], qubits
        )
    )

    for pauli_exp_box in pauli_exp_boxes[:-1]:
        circ.add_pauliexpbox(pauliexpbox=pauli_exp_box, qubits=reg_trotterbox.to_list())
    circ.add_pauliexpbox(
        pauliexpbox=pauli_exp_boxes[-1], qubits=reg_trotterbox.to_list()
    )
    for pauli_exp_box in pauli_exp_boxes[-2::-1]:
        circ.add_pauliexpbox(pauliexpbox=pauli_exp_box, qubits=reg_trotterbox.to_list())

    box = CircBox(circ)

    return box


def get_dilation_operators(epsilon, ancilla_qubits, system_qubits):
    val = np.sqrt(2 * epsilon) / 2
    K1_dict = {}
    qps = QubitPauliString([ancilla_qubits[0], system_qubits[0]], [Pauli.X, Pauli.X])
    K1_dict[qps] = val
    qps = QubitPauliString([ancilla_qubits[0], system_qubits[0]], [Pauli.Y, Pauli.Y])
    K1_dict[qps] = val
    K1 = QubitPauliOperator(K1_dict)

    K2_dict = {}
    qps = QubitPauliString([ancilla_qubits[0], system_qubits[-1]], [Pauli.X, Pauli.X])
    K2_dict[qps] = val
    qps = QubitPauliString([ancilla_qubits[0], system_qubits[-1]], [Pauli.Y, Pauli.Y])
    K2_dict[qps] = -val
    K2 = QubitPauliOperator(K2_dict)

    return [K1, K2]


def get_dilation_angles(epsilon, Delta_t):
    val = (2 / np.pi) * (np.sqrt(2 * epsilon) / 2) * np.sqrt(Delta_t)
    angles_K1 = [
        val,
        val,
        0.0,
    ]
    angles_K2 = [
        val,
        -val,
        0.0,
    ]
    dilation_angles = {
        "K1": angles_K1,
        "K2": angles_K2,
    }
    return dilation_angles


def get_dilation_step_deterministic(dilation_angles: dict, n_qubits: int):
    circ = Circuit(name="Dilation Trotter step")
    qreg_anc = circ.add_q_register("anc_dilation", 2)
    qreg_sys = circ.add_q_register("sys", n_qubits)

    circ.TK2(*dilation_angles["K1"], qreg_anc[0], qreg_sys[0])
    circ.TK2(*dilation_angles["K2"], qreg_anc[1], qreg_sys[n_qubits - 1])
    box = CircBox(circ)

    return box


def get_dissipator_simulation_circbox_deterministic(
    epsilon: float, n_system_qubits: int, t_dissipator: float, n_trotter_steps: int
):
    trotter_step_size = t_dissipator / n_trotter_steps

    dilation_angles = get_dilation_angles(epsilon, Delta_t=trotter_step_size)
    dilation_trotter_step = get_dilation_step_deterministic(
        dilation_angles, n_system_qubits
    )

    circ = Circuit(name="Dissipator Trotterization")
    anc_reg = circ.add_q_register("ancilla", 2)
    sys_reg = circ.add_q_register("system", n_system_qubits)

    for _ in range(n_trotter_steps):
        circ.add_circbox(dilation_trotter_step, anc_reg.to_list() + sys_reg.to_list())
    circ.Reset(anc_reg[0])
    circ.Reset(anc_reg[1])

    return CircBox(circ)


def get_dissipator_simulation_circbox(
    dilation_operator: QubitPauliOperator,
    system_qubits: list,
    ancilla_qubits: list,
    t_dissipator: float,
    n_trotter_steps: int,
):
    trotter_step_size = t_dissipator / n_trotter_steps

    dilation_trotter_step = (
        get_second_order_dilation_trotter_step_from_QubitPauliOperator(
            qubit_pauli_operator=dilation_operator,
            Delta_t=trotter_step_size,
            qubits=ancilla_qubits + system_qubits,
        )
    )

    circ = Circuit(name="Dissipator Trotterization")
    anc_reg = circ.add_q_register("ancilla", len(ancilla_qubits))
    sys_reg = circ.add_q_register("system", len(system_qubits))

    for _ in range(n_trotter_steps):
        circ.add_circbox(dilation_trotter_step, anc_reg.to_list() + sys_reg.to_list())
    circ.Reset(anc_reg.to_list()[0])

    return CircBox(circ)


def XXZ_lindblad_simulation_circ(
    n_system_qubits: int,
    Delta_ZZ: float,
    epsilon: float,
    t_evolution: float,
    n_evolution_steps: int,
):
    delta_t_evolution = t_evolution / n_evolution_steps
    n_ancilla_qubits = 2

    xxz_hamiltonian = get_xxz_chain_hamiltonian(
        n_qubits=n_system_qubits, Delta_ZZ=Delta_ZZ
    )

    coherent_box = get_hamiltonian_simulation_circbox(
        xxz_hamiltonian,
        n_system_qubits,
        t_trotterization=delta_t_evolution,
        n_trotter_steps=1,
    )
    dissipative_box_deterministic = get_dissipator_simulation_circbox_deterministic(
        epsilon,
        n_system_qubits=n_system_qubits,
        t_dissipator=delta_t_evolution,
        n_trotter_steps=1,
    )

    circ = Circuit()
    anc_reg = circ.add_q_register("ancilla", n_ancilla_qubits)
    sys_reg = circ.add_q_register("system", n_system_qubits)
    for _ in range(n_evolution_steps):
        circ.add_circbox(coherent_box, sys_reg.to_list())
        circ.add_circbox(
            dissipative_box_deterministic, anc_reg.to_list() + sys_reg.to_list()
        )

    return circ
