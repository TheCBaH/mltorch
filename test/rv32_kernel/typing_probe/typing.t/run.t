Register classes and immediate domains are distinct types: what the type
checker must reject. Every negative case is paired with a control that must
still compile.

  $ R=../../../..
  $ check() { sh $R/test/rv32_kernel/typing_probe/typing.sh $R/test/rv32_kernel/typing_probe/rv32_kernel_probe.exe "$1" "$2"; }

An integer register where a floating-point scalar is wanted.

  $ check "integer scalar for vfmacc.vf" "Insn.Vfmacc_vf { acc = Vreg.v8; scalar = Gpr.a4; src = Vreg.v0 }"
  integer scalar for vfmacc.vf: rejected
  $ check "float scalar for vfmacc.vf" "Insn.Vfmacc_vf { acc = Vreg.v8; scalar = Fpr.fa4; src = Vreg.v0 }"
  float scalar for vfmacc.vf: COMPILES

A floating-point register where an integer one is wanted.

  $ check "float scalar for vmacc.vx" "Insn.Vmacc_vx { acc = Vreg.v8; scalar = Fpr.fa4; src = Vreg.v0 }"
  float scalar for vmacc.vx: rejected
  $ check "integer scalar for vmacc.vx" "Insn.Vmacc_vx { acc = Vreg.v8; scalar = Gpr.a4; src = Vreg.v0 }"
  integer scalar for vmacc.vx: COMPILES

An immediate of another domain.

  $ check "12-bit immediate as a shift amount" "Insn.Srli { dst = Gpr.t3; src = Gpr.a4; shamt = Imm.Imm12.of_int 8 }"
  12-bit immediate as a shift amount: rejected
  $ check "shift amount as a shift amount" "Insn.Srli { dst = Gpr.t3; src = Gpr.a4; shamt = Imm.Shamt.of_int 8 }"
  shift amount as a shift amount: COMPILES

A register number only through the checked constructor.

  $ check "unchecked register constructor" "Gpr.of_int 3"
  unchecked register constructor: rejected
  $ check "checked register constructor" "Gpr.of_index 3"
  checked register constructor: COMPILES
  $ check "register is not an int" "(Gpr.a0 : int)"
  register is not an int: rejected
  $ check "register index is an int" "(Gpr.index Gpr.a0 : int)"
  register index is an int: COMPILES
