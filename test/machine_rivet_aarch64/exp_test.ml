(* The project-owned exp as AArch64 code, loaded and run natively over a batch
   of inputs: every result equal, bit for bit, to the specification, which
   test/machine_ir holds equal to the host libm. *)

open Machine_ir
module Exp = Machine_rivet_aarch64.Rivet_a64_exp
module Image = Machine_rivet_aarch64.Rivet_a64_image
module F = Machine_rivet_aarch64.Rivet_a64_form
module A = Aarch64
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive

let insn i = N.Instruction { insn = i; origin = F.origin }
let dir directive = N.Directive { directive; origin = F.origin }
let lbl name = N.Label { name; origin = F.origin }

(* [drive(io)]: the word count at io[0], then that many doubles, each replaced
   by its exp. *)
let driver =
  let i op ops = insn (F.ins op ops) in
  let x n = A.Operand.Reg { A.Reg.num = n; width = 64; is_sp = false } in
  let d n = A.Operand.Freg { A.Freg.num = n; double = true } in
  let sp_slot off =
    A.Operand.Mem
      {
        A.Mem.base = { A.Reg.num = 31; width = 64; is_sp = true };
        offset = A.Disp.Const (Int64.of_int off);
        writeback = false;
        pre = true;
      }
  in
  let sp = A.Operand.Reg { A.Reg.num = 31; width = 64; is_sp = true } in
  let at base off =
    A.Operand.Mem
      {
        A.Mem.base = { A.Reg.num = base; width = 64; is_sp = false };
        offset = A.Disp.Const (Int64.of_int off);
        writeback = false;
        pre = true;
      }
  in
  let sym n = A.Operand.Sym (Asm_core.Expr.Symbol n) in
  {
    N.unit_name = "driver";
    items =
      [
        dir
          (D.Section
             { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
        dir (D.Align { boundary = 4 });
        dir (D.Global { name = "drive" });
        dir (D.Sym_type { name = "drive"; kind = D.Function });
        lbl "drive";
        i A.Opcode.Sub [ sp; sp; F.imm 32L ];
        i A.Opcode.Str [ x 30; sp_slot 0 ];
        i A.Opcode.Str [ x 19; sp_slot 8 ];
        i A.Opcode.Str [ x 20; sp_slot 16 ];
        i A.Opcode.Mov [ x 19; x 0 ];
        i A.Opcode.Ldr [ x 20; at 19 0 ];
        i A.Opcode.Add [ x 19; x 19; F.imm 8L ];
        lbl "again";
        i A.Opcode.Cbz [ x 20; sym "finished" ];
        i A.Opcode.Ldr [ d 0; at 19 0 ];
        i A.Opcode.Bl [ sym Exp.symbol ];
        i A.Opcode.Str [ d 0; at 19 0 ];
        i A.Opcode.Add [ x 19; x 19; F.imm 8L ];
        i A.Opcode.Sub [ x 20; x 20; F.imm 1L ];
        i A.Opcode.B [ sym "again" ];
        lbl "finished";
        i A.Opcode.Ldr [ x 20; sp_slot 16 ];
        i A.Opcode.Ldr [ x 19; sp_slot 8 ];
        i A.Opcode.Ldr [ x 30; sp_slot 0 ];
        i A.Opcode.Add [ sp; sp; F.imm 32L ];
        i A.Opcode.Movz [ x 0; F.imm 0L ];
        i A.Opcode.Ret [];
        dir (D.Declared_section { name = ".note.GNU-stack" });
      ];
  }

let run inputs =
  let ( let* ) = Result.bind in
  let render e = Fmt.str "%a" Image.Error.pp e in
  let* exp =
    Result.map_error
      (fun r -> Fmt.str "exp: %a" Machine_rivet_aarch64.Rivet_a64_refusal.pp r)
      (Err.payload Exp.module_)
  in
  let* laid =
    Result.map_error render
      (Err.payload (Image.plan ~entry:"drive" [ driver; exp ]))
  in
  let* loaded = Result.map_error render (Err.payload (Image.load laid)) in
  Fun.protect
    ~finally:(fun () -> Image.close loaded)
    (fun () ->
      let n = Array.length inputs in
      let io =
        Bigarray.Array1.create Bigarray.char Bigarray.c_layout (8 * (n + 1))
      in
      let put k v =
        for b = 0 to 7 do
          Bigarray.Array1.set io
            ((8 * k) + b)
            (Char.chr
               (Int64.to_int
                  (Int64.logand (Int64.shift_right_logical v (8 * b)) 0xffL)))
        done
      in
      let get k =
        let v = ref 0L in
        for b = 7 downto 0 do
          v :=
            Int64.logor (Int64.shift_left !v 8)
              (Int64.of_int (Char.code (Bigarray.Array1.get io ((8 * k) + b))))
        done;
        !v
      in
      put 0 (Int64.of_int n);
      Array.iteri (fun k x -> put (k + 1) (Int64.bits_of_float x)) inputs;
      let* _ = Result.map_error render (Err.payload (Image.call ~io loaded)) in
      Ok (Array.init n (fun k -> get (k + 1))))

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

let ties =
  Array.of_list
    (List.concat_map
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
       (List.init 2001 (fun k -> k - 1000)))

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

let%expect_test "owned exp, native AArch64" =
  differences "specials" specials;
  differences "unit interval" (uniform 1 (-1.) 1. 200_000);
  differences "model range" (uniform 2 (-20.) 20. 400_000);
  differences "wide" (uniform 3 (-745.2) 709.8 400_000);
  differences "special-case band" (uniform 4 (-1024.) 1024. 200_000);
  differences "half-integer reductions" ties;
  differences "binade edges" (Array.of_list binade_edges);
  differences "bit patterns" (patterns 5 400_000);
  [%expect
    {|
    specials: 29 inputs, 0 differences
    unit interval: 200000 inputs, 0 differences
    model range: 400000 inputs, 0 differences
    wide: 400000 inputs, 0 differences
    special-case band: 200000 inputs, 0 differences
    half-integer reductions: 2015 inputs, 0 differences
    binade edges: 14400 inputs, 0 differences
    bit patterns: 400000 inputs, 0 differences |}]

(* GNU's assembler and linker, on the printed module, agree with Rivet's bytes
   and symbols. *)
let%expect_test "owned exp, GNU coherence" =
  (match Err.payload Exp.module_ with
  | Error r ->
      Fmt.pr "module: %a@." Machine_rivet_aarch64.Rivet_a64_refusal.pp r
  | Ok m ->
      Fmt.pr "%a@." Machine_rivet_aarch64_gnu.Gnu_coherence.Verdict.pp
        (Machine_rivet_aarch64_gnu.Gnu_coherence.check ~entry:Exp.symbol [ m ]));
  [%expect {| agree (2 segments, 2508 bytes, 2 symbols) |}]
