(* AArch64 text as CompCert 3.17 prints it for a double and a float function:
   [ex] mixes both precisions, [fsum] is binary32 only. *)
let ex =
  {|	.text
	.balign 4
	.globl ex
ex:
	.cfi_startproc
	mov	x15, sp
	stp	x15, x30, [sp, #-16]!
	fadd	d4, d0, d0
	fcvt	d5, s1
	fadd	d2, d4, d5
	fcvt	s0, d2
	fadd	s7, s0, s1
	fmov	d1, #1.0000000
	fcmp	d2, d1
	b.le	.L100
	scvtf	d3, x0
	fdiv	d0, d2, d3
	b	.L101
.L100:
	fcvt	d0, s7
.L101:
	ldr	x30, [sp, #8]
	ret	x30
|}

let fsum =
  {|fsum:
	fmov	s0, wzr
	movz	w2, #0, lsl #0
.L102:
	cmp	w2, w1
	b.ge	.L103
	ldr	s1, [x0, w2, sxtw #2]
	fadd	s0, s0, s1
	add	w2, w2, #1
	b	.L102
.L103:
	ldr	x30, [sp, #8]
	ret	x30
|}

let census isa text =
  match Err.payload (Asm_census.of_text isa text) with
  | Ok t ->
      Fmt.str "%a%a" Asm_census.pp t
        Fmt.(list ~sep:nop (fun ppf (m, n) -> pf ppf "; %s x%d" m n))
        (Asm_census.mnemonics t)
  | Error e -> Fmt.str "error: %a" Asm_census.pp_error e

let%expect_test "aarch64: CompCert output" =
  print_endline (census Asm_census.Isa.Aarch64 ex);
  print_endline (census Asm_census.Isa.Aarch64 fsum);
  [%expect
    {|
    arithmetic 4, conversion 4, memory_move 1; fadd x2; fcmp x1; fcvt x3; fdiv x1; fmov x1; scvtf x1
    arithmetic 0, conversion 0, memory_move 0 |}]

(* The check can fail: one binary64 add in an otherwise binary32 listing is
   counted, and an unclassified binary64 mnemonic is an error. *)
let%expect_test "aarch64: a single binary64 operation is seen" =
  print_endline (census Asm_census.Isa.Aarch64 (fsum ^ "\tfadd\td0, d0, d1\n"));
  print_endline (census Asm_census.Isa.Aarch64 (fsum ^ "\tfzzz\td0, d0, d1\n"));
  print_endline
    (census Asm_census.Isa.Aarch64
       "\tldr\td1, [x0, #8]\n\tstr\td1, [x1]\n\tmov\tv0.2d, v1.2d\n");
  [%expect
    {|
    arithmetic 1, conversion 0, memory_move 0; fadd x1
    error: aarch64: floating-point mnemonic "fzzz" is in no census table
    arithmetic 0, conversion 0, memory_move 3; ldr x1; mov x1; str x1 |}]

(* x86-64 as gas prints it for the same two functions. *)
let x64_ex =
  {|	.text
ex:
	addsd	%xmm0, %xmm0
	cvtss2sd	%xmm1, %xmm5
	cvtsd2ss	%xmm0, %xmm0
	addss	%xmm1, %xmm0
	movsd	.LC0(%rip), %xmm1
	ucomisd	%xmm1, %xmm0
	cvtsi2sdq	%rdi, %xmm3
	divsd	%xmm3, %xmm0
	movq	%xmm0, %rax
	movq	%rax, %rbx
	ret
|}

let x64_fsum =
  {|fsum:
	xorps	%xmm0, %xmm0
.L1:
	movss	(%rdi), %xmm1
	addss	%xmm1, %xmm0
	cvttss2sil	%xmm0, %eax
	movd	%xmm0, %eax
	ret
|}

let%expect_test "x86_64" =
  print_endline (census Asm_census.Isa.X86_64 x64_ex);
  print_endline (census Asm_census.Isa.X86_64 x64_fsum);
  print_endline
    (census Asm_census.Isa.X86_64 (x64_fsum ^ "\tmulsd\t%xmm1, %xmm0\n"));
  print_endline (census Asm_census.Isa.X86_64 "\tfoobar\t%xmm0, %xmm1\n");
  [%expect
    {|
    arithmetic 3, conversion 3, memory_move 2; addsd x1; cvtsd2ss x1; cvtsi2sdq x1; cvtss2sd x1; divsd x1; movq x1; movsd x1; ucomisd x1
    arithmetic 0, conversion 0, memory_move 0
    arithmetic 1, conversion 0, memory_move 0; mulsd x1
    error: x86_64: floating-point mnemonic "foobar" is in no census table |}]
