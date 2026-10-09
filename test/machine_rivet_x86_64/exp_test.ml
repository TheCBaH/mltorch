(* The project-owned exp as x86-64 code, run under qemu-user over a batch of
   inputs: every result equal, bit for bit, to the specification, which
   test/machine_ir holds equal to the host libm. Emulation, not an x86-64 CPU. *)

open Machine_ir
module Exp = Machine_rivet_x86_64.Rivet_x64_exp
module Q = Machine_rivet_x86_64.Rivet_x64_qemu
module F = Machine_rivet_x86_64.Rivet_x64_form
module Fam = X86_family_encode
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive

let insn i = N.Instruction { insn = i; origin = F.origin }
let dir directive = N.Directive { directive; origin = F.origin }
let lbl name = N.Label { name; origin = F.origin }

(* [_start]: for each input word, call exp and store the result in place; then
   write the whole block to standard output. *)
let driver (inputs : float array) =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      F.mutation = None;
      esc;
      reference = (fun _ -> None);
      table_slot = (fun _ -> None);
    }
  in
  let reg n = Fam.Operand.Reg (F.find env n) in
  let i mnemonic ops = insn (F.make env mnemonic ops) in
  let sym n = Fam.Operand.Sym (Asm_core.Expr.Symbol n) in
  let block =
    Fam.Operand.Mem
      {
        Fam.Mem.base = Some Fam.rip_reg;
        index = None;
        scale = 1;
        disp = Fam.Disp.Sym (Asm_core.Expr.Symbol "inputs");
      }
  in
  let n = Array.length inputs in
  let at off = F.mem_op ~base:(F.find env "r12") ~disp:off () in
  let text =
    [
      dir
        (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = "_start" });
      lbl "_start";
      i "leaq" [ block; reg "r12" ];
      i "movq" [ F.imm (Int64.of_int n); reg "r13" ];
      lbl "again";
      i "movsd" [ at 0L; reg "xmm0" ];
      i "call" [ sym Exp.symbol ];
      i "movsd" [ reg "xmm0"; at 0L ];
      i "addq" [ F.imm 8L; reg "r12" ];
      i "subq" [ F.imm 1L; reg "r13" ];
      i "jne" [ sym "again" ];
      i "movl" [ F.imm 1L; reg "eax" ];
      i "movl" [ F.imm 1L; reg "edi" ];
      i "leaq" [ block; reg "rsi" ];
      i "movl" [ F.imm (Int64.of_int (8 * n)); reg "edx" ];
      i "syscall" [];
      i "movl" [ F.imm 60L; reg "eax" ];
      i "xorl" [ reg "edi"; reg "edi" ];
      i "syscall" [];
    ]
  in
  let data =
    [
      dir
        (D.Section { name = ".data"; perms = Asm_core.Perms.rw; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = "inputs" });
      lbl "inputs";
      dir
        (D.Data
           {
             width = 8;
             values =
               Array.to_list
                 (Array.map
                    (fun x ->
                      Asm_core.Expr.Const
                        (Foundation.Bigint.of_int64 (Int64.bits_of_float x)))
                    inputs);
           });
    ]
  in
  {
    N.unit_name = "driver";
    items =
      text @ data @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }

let run inputs =
  let ( let* ) = Result.bind in
  let payload what r =
    Result.map_error
      (fun e ->
        Fmt.str "%s: %a" what Machine_rivet_x86_64.Rivet_x64_refusal.pp e)
      (Err.payload r)
  in
  let* exp = payload "exp" Exp.module_ in
  let* drv = payload "driver" (driver inputs) in
  let* elf = Q.elf_of ~entry:"_start" [ drv; exp ] in
  let* out = Q.execute elf in
  if String.length out <> 8 * Array.length inputs then
    Error (Fmt.str "wrote %d bytes" (String.length out))
  else
    Ok
      (Array.init (Array.length inputs) (fun k ->
           String.get_int64_le out (8 * k)))

let same a b =
  if Float.is_nan (Int64.float_of_bits a) then
    Float.is_nan (Int64.float_of_bits b)
  else Int64.equal a b

let differences name inputs =
  match run inputs with
  | Error e -> Fmt.pr "%s: %s@." name e
  | Ok got ->
      let bad = ref 0 and first = ref None in
      Array.iteri
        (fun k x ->
          let want = Int64.bits_of_float (Mir_exp.model x) in
          if not (same got.(k) want) then begin
            incr bad;
            if !first = None then first := Some (x, got.(k), want)
          end)
        inputs;
      Fmt.pr "%s: %d inputs, %d differences@." name (Array.length inputs) !bad;
      Option.iter
        (fun (x, g, w) -> Fmt.pr "  first: %h gave %Lx, wanted %Lx@." x g w)
        !first

let specials =
  [|
    0.;
    -0.;
    1.;
    -1.;
    Float.nan;
    Float.infinity;
    Float.neg_infinity;
    0x1p-54;
    0x1p-55;
    -0x1p-54;
    Float.min_float;
    5e-324;
    511.99999999999994;
    512.;
    -512.;
    709.782712893384;
    709.782712893385;
    710.;
    1023.9999999999999;
    1024.;
    -745.1332191019411;
    -745.1332191019412;
    -708.3964185322641;
    -708.4;
    -1023.9999999999999;
    -1024.;
    1e300;
    -1e300;
    Float.max_float;
  |]

(* Inputs whose scaled value lands exactly on a half integer, where rounding
   half away from zero and half to even choose different reduction steps. *)
let ties =
  List.concat_map
    (fun k ->
      let x0 = (float_of_int k +. 0.5) /. Mir_exp.invln2n in
      List.filter
        (fun x ->
          let z = x *. Mir_exp.invln2n in
          Float.abs (z -. Float.trunc z) = 0.5)
        [
          Float.pred (Float.pred x0);
          Float.pred x0;
          x0;
          Float.succ x0;
          Float.succ (Float.succ x0);
        ])
    (List.init 2001 (fun k -> k - 1000))

(* Inputs whose scaled value falls just short of a half integer below a power
   of two, where adding one half would round up to the wrong integer. *)
let binade_edges =
  List.concat_map
    (fun n ->
      List.concat_map
        (fun sign ->
          let x0 = sign *. (Float.ldexp 1. n -. 0.5) /. Mir_exp.invln2n in
          let rec walk k x acc =
            if k = 0 then acc else walk (k - 1) (Float.pred x) (x :: acc)
          in
          let rec climb k x =
            if k = 0 then x else climb (k - 1) (Float.succ x)
          in
          walk 400 (climb 200 x0) [])
        [ 1.; -1. ])
    (List.init 18 Fun.id)

let uniform seed lo hi n =
  let st = Random.State.make [| seed |] in
  Array.init n (fun _ -> lo +. Random.State.float st (hi -. lo))

let patterns seed n =
  let st = Random.State.make [| seed |] in
  Array.init n (fun _ -> Int64.float_of_bits (Random.State.bits64 st))

let%expect_test "owned exp, emulated x86-64" =
  differences "specials" specials;
  differences "unit interval" (uniform 1 (-1.) 1. 20_000);
  differences "model range" (uniform 2 (-20.) 20. 40_000);
  differences "wide" (uniform 3 (-745.2) 709.8 40_000);
  differences "special-case band" (uniform 4 (-1024.) 1024. 20_000);
  differences "half-integer reductions" (Array.of_list ties);
  differences "binade edges" (Array.of_list binade_edges);
  differences "bit patterns" (patterns 5 40_000);
  [%expect
    {|
    specials: 29 inputs, 0 differences
    unit interval: 20000 inputs, 0 differences
    model range: 40000 inputs, 0 differences
    wide: 40000 inputs, 0 differences
    special-case band: 20000 inputs, 0 differences
    half-integer reductions: 2015 inputs, 0 differences
    binade edges: 14400 inputs, 0 differences
    bit patterns: 40000 inputs, 0 differences |}]

(* GNU's assembler and linker, on the printed module, agree with Rivet's bytes
   and symbols. *)
let%expect_test "owned exp, GNU coherence" =
  (match Err.payload Exp.module_ with
  | Error r -> Fmt.pr "module: %a@." Machine_rivet_x86_64.Rivet_x64_refusal.pp r
  | Ok m ->
      Fmt.pr "%a@." Machine_rivet_x86_64_gnu.Gnu_coherence.Verdict.pp
        (Machine_rivet_x86_64_gnu.Gnu_coherence.check ~entry:Exp.symbol [ m ]));
  [%expect {| agree (2 segments, 2616 bytes, 2 symbols) |}]
