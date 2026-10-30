(* Two fused-attention functions that do not fit the instruction memory
   together, on the subsystem. Resident, the image is refused at the ITCM; as
   overlays, each loaded on demand by the DMA into one slot and its bss in one
   shared DTCM region, both outputs are bitwise the streaming reference's. *)

module A = Attention_block
module S = A.Strides
module Z = Coralnpu_kernels_zvt
module Asm = Machine_rivet_riscv32.Rv32_asm
module L = Coral_layout

let dir = Sys.getenv "CORALNPU_SUBSYSTEM_DATA"
let rv32_profile = Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_ilp32f
let entry = "overlay_caller"
let or_fail pp = function Ok v -> v | Error e -> Fmt.failwith "%a" pp e

(* One attention over bfloat16 [tq] x [d] queries and [tk] keys, no mask. *)
let make ~tq ~tk ~d ~bk =
  let i64 = Int64.of_int in
  let strides row = { S.batch = 0L; head = 0L; row = i64 row; col = 1L } in
  Result.get_ok
    (Err.payload
       (A.make ~batch:1L ~heads:1L ~tq:(i64 tq) ~tk:(i64 tk) ~d:(i64 d)
          ~q:(strides d) ~k:(strides d) ~v:(strides d) ~o:(strides d)
          ~scale:(A.Scale.default ~d:(i64 d))
          ~block_keys:(i64 bk) ()))

let lcg = ref 12345

let bf16_value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  let n = !lcg in
  Half.Bf16.of_float (float_of_int ((n / 7 mod 33) - 16) /. 8.)

let bf16_bytes xs =
  let b = Bytes.create (2 * Array.length xs) in
  Array.iteri (fun i x -> Bytes.set_uint16_le b (2 * i) x) xs;
  Bytes.to_string b

let f32_bytes xs =
  let b = Bytes.create (4 * Array.length xs) in
  Array.iteri
    (fun i x -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float x))
    xs;
  Bytes.to_string b

(* The operands as bytes and the reference's output. *)
let case t =
  let n = Int64.to_int in
  let tq = n t.A.tq and tk = n t.A.tk and d = n t.A.d in
  let operand rows = Array.init (rows * d) (fun _ -> bf16_value ()) in
  let q = operand tq and k = operand tk and v = operand tk in
  let out = Array.make (tq * d) 0. in
  let at xs o = Half.Bf16.to_float xs.(n o) in
  A.eval t ~q:(at q) ~k:(at k) ~v:(at v)
    ~mask:(fun _ -> 0.)
    ~write:(fun o x -> out.(n o) <- x);
  ([ bf16_bytes q; bf16_bytes k; bf16_bytes v ], f32_bytes out)

let shapes =
  [ make ~tq:32 ~tk:32 ~d:32 ~bk:32; make ~tq:16 ~tk:32 ~d:96 ~bk:16 ]

(* Calls each function with its operands' slots, then its output's. *)
let caller =
  let i = Asm.insn and r = Asm.reg in
  Result.get_ok
    (Asm.module_ "overlay_caller" (fun () ->
         [
           Asm.text;
           Asm.align 4;
           Asm.global entry;
           Asm.label entry;
           i "addi" [ r "sp"; r "sp"; Asm.imm (-16) ];
           i "sw" [ r "ra"; Asm.mem "sp" 12 ];
           i "mv" [ r "s0"; r "a0" ];
         ]
         @ List.concat
             (List.mapi
                (fun j t ->
                  let slot k = Asm.mem "s0" (4 * k) in
                  [
                    i "lw" [ r "a0"; slot (3 * j) ];
                    i "lw" [ r "a1"; slot ((3 * j) + 1) ];
                    i "lw" [ r "a2"; slot ((3 * j) + 2) ];
                    i "lw" [ r "a3"; slot ((3 * List.length shapes) + j) ];
                    i "li" [ r "a4"; Asm.imm 0 ];
                    i "call" [ Asm.sym (A.symbol t) ];
                  ])
                shapes)
         @ [
             i "lw" [ r "ra"; Asm.mem "sp" 12 ];
             i "addi" [ r "sp"; r "sp"; Asm.imm 16 ];
             i "li" [ r "a0"; Asm.imm 0 ];
             i "ret" [];
           ]))

let functions =
  List.map
    (fun t -> (A.symbol t, or_fail Fmt.string (Z.Zvt_sdpa.module_ t)))
    shapes

let cases = List.map case shapes

let slots =
  let slot placement size =
    L.Slot.make ~home:L.Home.Ram ~align:16L placement (Int64.of_int size)
  in
  List.concat_map
    (fun (operands, _) ->
      List.map (fun b -> slot L.Placement.In (String.length b)) operands)
    cases
  @ List.map (fun (_, out) -> slot L.Placement.Out (String.length out)) cases

let profile =
  or_fail Coral_profile.pp_error
    (Err.payload (Coral_profile.load ~dir "subsystem-matrix-normal"))

let bound =
  let operands = List.concat_map fst cases in
  fun k -> List.nth_opt operands k

let params = { L.Params.stack = 2048L; staging = 0L }

let build ~overlays =
  Coral_image.build ~params ~rv32_profile ~matrix:true
    ~overlays:
      (if overlays then
         List.map
           (fun (symbol, m) ->
             {
               Coral_overlay.Overlay.name = symbol;
               entry = symbol;
               modules = [ m ];
             })
           functions
       else [])
    profile ~entry
    ~kernel:(caller :: (if overlays then [] else List.map snd functions))
    slots

let%expect_test "resident, the image is refused at the instruction memory" =
  (match Err.payload (build ~overlays:false) with
  | Ok _ -> Fmt.pr "built@."
  | Error e -> Fmt.pr "refused: %a@." Coral_image.pp_error e);
  [%expect {| refused: itcm: 9420 bytes needed, 8192 available |}]

let%expect_test "as overlays, both attentions run bitwise" =
  let image =
    or_fail Coral_image.pp_error (Err.payload (build ~overlays:true))
  in
  let n_in = 3 * List.length shapes in
  let outputs = List.mapi (fun j _ -> n_in + j) shapes in
  (match
     Err.payload
       (Coral_soc_run.run ~deadline:3600. ~read_back:outputs ~dir image ~bound)
   with
  | Error e -> Fmt.pr "run: %a@." Coral_soc_run.Run_error.pp e
  | Ok o ->
      List.iteri
        (fun j (_, expected) ->
          let got = List.assoc (n_in + j) o.Coral_soc_run.read_back in
          Fmt.pr "attention %d: %s@." j
            (if got = expected then "bitwise" else "DIFFERS"))
        cases;
      Fmt.pr "%Ld cycles@." o.Coral_soc_run.mcycle);
  [%expect
    {|
    attention 0: bitwise
    attention 1: bitwise
    187103 cycles |}]
