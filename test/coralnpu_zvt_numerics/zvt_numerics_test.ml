(* [C <- A1 B1^T; C <- C + A2 B2^T] in one 16x16 binary32 accumulator tile,
   two [vtfmm] steps of tk = 1, on the VME core: each element against the
   fused step ([fmaf (a2, b2, c)]) and the separately rounded one
   ([c + round (a2 b2)]), for either orientation of the tile. Rows (A's
   elements) and columns (B's) are built so that some elements tell the two
   apart (a binary64 sum landing on a binary32 midpoint), some are random,
   and one row starts from zero to show what a subnormal product becomes. *)

open Peak_kernels
module L = Coral_layout

let n = 16
let u = ldexp 1. (-11)
let r32 = F32_math.round32

(* Deterministic binary32 values with mixed magnitudes. *)
let rand seed k =
  let h = ((seed * 7919) + (k * 104729)) * 2654435761 land 0x3fffffff in
  r32
    (float_of_int ((h mod 2001) - 1000) /. float_of_int ((h / 2001 mod 97) + 3))

let a1 = Array.init n (fun i -> if i = 15 then 0. else 1.)
let b1 = Array.make n 1.

let a2 =
  Array.init n (fun i ->
      if i = 15 then r32 1e-20 else if i < 8 then 1. +. u else rand 1 i)

let b2 =
  Array.init n (fun j ->
      if j = 15 then r32 1e-20
      else if j < 8 then ldexp 1. (-24) *. (1. -. u +. (u *. u))
      else rand 2 j)

let bytes_of arrays =
  let b = Bytes.create (4 * n * List.length arrays) in
  List.iteri
    (fun s arr ->
      Array.iteri
        (fun k x ->
          Bytes.set_int32_le b (4 * ((s * n) + k)) (Int32.bits_of_float x))
        arr)
    arrays;
  Bytes.to_string b

let kernel () =
  let open Err.Syntax in
  let v n = Reg.Vreg.of_index n in
  let* v8 = v 8 in
  let* v16 = v 16 in
  let* v0 = v 0 in
  let* v24 = v 24 in
  let* config =
    Peak_zvt.Tile.configure
      ~mtype:(Peak_zvt.Tile.mtype ~tk:1 ~mtwiden:1)
      ~vtype:0xD2
  in
  let* store =
    Peak_zvt.Tile.store ~base:Gpr.a6 ~stride:(4 * n) ~rows:n 0 Reg.Mtreg.mt0
  in
  let load dst off = [ addi Gpr.a4 Gpr.a5 off; vle Vtype.Sew.E32 dst Gpr.a4 ] in
  Ok
    (start_items
    @ [ lw Gpr.a5 Gpr.a0 0; lw Gpr.a6 Gpr.a0 4 ]
    @ config @ load v8 0 @ load v16 64 @ load v0 128 @ load v24 192
    @ [
        Item.insn (Vtzero { dst = Reg.Mtreg.mt0 });
        Item.insn
          (Vtmm
             { op = Insn.Tile_op.Vtfmm; acc = Reg.Mtreg.mt0; a = v8; b = v16 });
        Item.insn
          (Vtmm
             { op = Insn.Tile_op.Vtfmm; acc = Reg.Mtreg.mt0; a = v0; b = v24 });
      ]
    @ config @ store
    @ [ li Gpr.a0 0; Item.insn Ret ])

let fused i j = F32_math.fma32 a2.(i) b2.(j) (r32 (a1.(i) *. b1.(j)))
let separate i j = r32 (r32 (a1.(i) *. b1.(j)) +. r32 (a2.(i) *. b2.(j)))

let%expect_test "vtfmm binary32 accumulation" =
  let dir = Sys.getenv "CORALNPU_DATA" in
  let k =
    {
      items = kernel;
      slots =
        [ slot L.Placement.In (4 * n * 4); slot L.Placement.Out (4 * n * n) ];
      inputs = [ bytes_of [ a1; b1; a2; b2 ] ];
      expected = "";
      shape = "";
      matrix = true;
    }
  in
  let fail what = Fmt.pr "%s@." what in
  match
    ( Err.payload (Coral_profile.load ~dir "vme-core-mini-axi"),
      build k rv32_profile )
  with
  | Error e, _ -> fail (Fmt.str "%a" Coral_profile.pp_error e)
  | _, Error e -> fail e
  | Ok profile, Ok m ->
      (match
         Err.payload
           (Coral_image.build ~rv32_profile ~matrix:true profile ~entry
              ~kernel:[ m ] k.slots)
       with
      | Error e -> fail (Fmt.str "%a" Coral_image.pp_error e)
      | Ok image -> (
          match
            Err.payload
              (Coral_run.run ~flags:[ "--cycles=2000000" ] ~dir image
                 ~bound:(fun s ->
                   if s = 0 then Some (List.hd k.inputs) else None))
          with
          | Error e -> fail (Fmt.str "%a" Coral_run.Run_error.pp e)
          | Ok o ->
              let out = List.assoc 1 o.Coral_run.outputs in
              let at r c =
                Int32.float_of_bits
                  (String.get_int32_le out (4 * ((r * n) + c)))
              in
              let same x y = Int32.bits_of_float x = Int32.bits_of_float y in
              let tally orient =
                let f = ref 0
                and s = ref 0
                and both = ref 0
                and neither = ref 0 in
                for i = 0 to n - 1 do
                  for j = 0 to n - 1 do
                    let x = if orient then at i j else at j i in
                    let fu = same x (fused i j)
                    and se = same x (separate i j) in
                    match (fu, se) with
                    | true, true -> incr both
                    | true, false -> incr f
                    | false, true -> incr s
                    | false, false -> incr neither
                  done
                done;
                (!f, !s, !both, !neither)
              in
              List.iter
                (fun (name, orient) ->
                  let f, s, b, x = tally orient in
                  Fmt.pr
                    "%s: fused only %d, separate only %d, both %d, neither %d@."
                    name f s b x)
                [ ("rows are A", true); ("rows are B", false) ];
              Fmt.pr "midpoint element: %h (fused %h, separate %h)@." (at 0 0)
                (fused 0 0) (separate 0 0);
              Fmt.pr "subnormal element: %h (fused %h)@." (at 15 15)
                (fused 15 15)));
      [%expect
        {|
    rows are A: fused only 0, separate only 73, both 183, neither 0
    rows are B: fused only 0, separate only 64, both 8, neither 184
    midpoint element: 0x1p+0 (fused 0x1.000002p+0, separate 0x1p+0)
    subnormal element: 0x1.16c2p-133 (fused 0x1.16c2p-133) |}]

(* The same question for bfloat16: [vtfmm.alt] at tk = 2 reduces two k
   slices an instruction, so how it rounds them into the binary32
   accumulator decides which numerical requirement it can serve. A first
   instruction sets every element to 1 (1 x 1 + 0 x 0); the second adds two
   products [p0] (slice 0) and [p1] (slice 1), each exact in binary32, under
   four candidates: sequential in slice order, sequential reversed, the two
   products summed first, and all three rounded once. Rows 0-3 against
   columns 0-3 make [p0 = p1 = 2^-24] (sequential ties twice to 1; the
   others reach 1 + 2^-23); rows 4-7 against columns 4-7 make [p0 = 2^-24],
   [p1 = 2^-48] (only the single rounding passes the midpoint); the rest
   are small exact bfloat16 integers over 16. *)
let bf16_bits x =
  Int32.to_int (Int32.shift_right_logical (Int32.bits_of_float x) 16)

let rand16 seed k =
  let h = ((seed * 7919) + (k * 104729)) * 2654435761 land 0x3fffffff in
  float_of_int ((h mod 255) - 127) /. 16.

(* slice [s] of A (rows) and of B (columns) of the second instruction *)
let alt_a s i =
  if i < 4 then ldexp 1. (-12)
  else if i < 8 then if s = 0 then ldexp 1. (-12) else ldexp 1. (-24)
  else rand16 (3 + s) i

let alt_b s j =
  if j < 4 then ldexp 1. (-12)
  else if j < 8 then if s = 0 then ldexp 1. (-12) else ldexp 1. (-24)
  else rand16 (5 + s) j

let bf16_bytes slices =
  let b = Bytes.create (2 * n * List.length slices) in
  List.iteri
    (fun s f ->
      for k = 0 to n - 1 do
        Bytes.set_uint16_le b (2 * ((s * n) + k)) (bf16_bits (f k))
      done)
    slices;
  Bytes.to_string b

let alt_kernel () =
  let open Err.Syntax in
  let v n = Reg.Vreg.of_index n in
  let* v8 = v 8 in
  let* v16 = v 16 in
  let* v0 = v 0 in
  let* v24 = v 24 in
  let* alt =
    Peak_zvt.Tile.configure
      ~mtype:(Peak_zvt.Tile.mtype ~tk:2 ~mtwiden:2)
      ~vtype:0x1C9
  in
  let* f32 =
    Peak_zvt.Tile.configure
      ~mtype:(Peak_zvt.Tile.mtype ~tk:1 ~mtwiden:1)
      ~vtype:0xD2
  in
  let* store =
    Peak_zvt.Tile.store ~base:Gpr.a6 ~stride:(4 * n) ~rows:n 1 Reg.Mtreg.mt0
  in
  (* an operand's two slices, 32 bytes apart in memory and, as the peak
     kernels place them, 2 x LMUL groups (four registers) apart *)
  let load index off =
    let* dst = v index in
    let* hi = v (index + 4) in
    Ok
      [
        addi Gpr.a4 Gpr.a5 off;
        vle Vtype.Sew.E16 dst Gpr.a4;
        addi Gpr.a4 Gpr.a5 (off + 32);
        vle Vtype.Sew.E16 hi Gpr.a4;
      ]
  in
  let* l8 = load 8 0 in
  let* l16 = load 16 64 in
  let* l0 = load 0 128 in
  let* l24 = load 24 192 in
  Ok
    (start_items
    @ [ lw Gpr.a5 Gpr.a0 0; lw Gpr.a6 Gpr.a0 4 ]
    @ alt @ l8 @ l16 @ l0 @ l24
    @ [
        Item.insn (Vtzero { dst = Reg.Mtreg.mt0 });
        Item.insn
          (Vtmm
             {
               op = Insn.Tile_op.Vtfmm_alt;
               acc = Reg.Mtreg.mt0;
               a = v8;
               b = v16;
             });
        Item.insn
          (Vtmm
             {
               op = Insn.Tile_op.Vtfmm_alt;
               acc = Reg.Mtreg.mt0;
               a = v0;
               b = v24;
             });
      ]
    @ f32 @ store
    @ [ li Gpr.a0 0; Item.insn Ret ])

let alt_models i j =
  let c = 1. in
  let p0 = alt_a 0 i *. alt_b 0 j and p1 = alt_a 1 i *. alt_b 1 j in
  [
    ("sequential", r32 (r32 (c +. r32 p0) +. r32 p1));
    ("reversed", r32 (r32 (c +. r32 p1) +. r32 p0));
    ("pair first", r32 (c +. r32 (p0 +. p1)));
    ("one rounding", r32 (c +. p0 +. p1));
  ]

let alt_run dir configuration =
  let k =
    {
      items = alt_kernel;
      slots =
        [ slot L.Placement.In (2 * n * 8); slot L.Placement.Out (4 * n * n) ];
      inputs =
        [
          bf16_bytes
            [
              (fun _ -> 1.);
              (fun _ -> 0.);
              (fun _ -> 1.);
              (fun _ -> 0.);
              alt_a 0;
              alt_a 1;
              alt_b 0;
              alt_b 1;
            ];
        ];
      expected = "";
      shape = "";
      matrix = true;
    }
  in
  let ( let* ) = Result.bind in
  let* profile =
    Result.map_error
      (Fmt.str "%a" Coral_profile.pp_error)
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let* m = build k rv32_profile in
  let* image =
    Result.map_error
      (Fmt.str "%a" Coral_image.pp_error)
      (Err.payload
         (Coral_image.build ~rv32_profile ~matrix:true profile ~entry
            ~kernel:[ m ] k.slots))
  in
  let* o =
    Result.map_error
      (Fmt.str "%a" Coral_run.Run_error.pp)
      (Err.payload
         (Coral_run.run ~flags:[ "--cycles=2000000" ] ~dir image
            ~bound:(fun s -> if s = 0 then Some (List.hd k.inputs) else None)))
  in
  Ok (List.assoc 1 o.Coral_run.outputs)

let%expect_test "vtfmm.alt bfloat16 accumulation" =
  match alt_run (Sys.getenv "CORALNPU_DATA") "vme-core-mini-axi" with
  | Error e -> Fmt.pr "%s@." e
  | Ok out ->
      let at r c =
        Int32.float_of_bits (String.get_int32_le out (4 * ((r * n) + c)))
      in
      let same x y = Int32.bits_of_float x = Int32.bits_of_float y in
      List.iter
        (fun (name, read, region) ->
          let counts = Hashtbl.create 4 in
          List.iter
            (fun (i, j) ->
              List.iter
                (fun (model, x) ->
                  if same (read i j) x then
                    Hashtbl.replace counts model
                      (1
                      + Option.value ~default:0 (Hashtbl.find_opt counts model)
                      ))
                (alt_models i j))
            region;
          Fmt.pr "%s (%d):%a@." name (List.length region)
            Fmt.(
              list ~sep:nop (fun ppf (m, _) ->
                  pf ppf " %s %d" m
                    (Option.value ~default:0 (Hashtbl.find_opt counts m))))
            (alt_models 0 0))
        (let cells f =
           List.concat_map
             (fun i ->
               List.filter_map
                 (fun j -> if f i j then Some (i, j) else None)
                 (List.init n Fun.id))
             (List.init n Fun.id)
         in
         [
           ("ties", at, cells (fun i j -> i < 4 && j < 4));
           ( "midpoint",
             at,
             cells (fun i j -> i >= 4 && i < 8 && j >= 4 && j < 8) );
           ("random, rows are A", at, cells (fun i j -> i >= 8 && j >= 8));
           ( "random, rows are B",
             (fun i j -> at j i),
             cells (fun i j -> i >= 8 && j >= 8) );
         ]);
      Fmt.pr
        "tie element %h, midpoint element %h; element (8, 9) %h (sequential \
         %h)@."
        (at 0 0) (at 4 4) (at 8 9)
        (List.assoc "sequential" (alt_models 8 9));
      [%expect
        {|
        ties (16): sequential 0 reversed 0 pair first 16 one rounding 16
        midpoint (16): sequential 0 reversed 0 pair first 0 one rounding 16
        random, rows are A (64): sequential 64 reversed 64 pair first 64 one rounding 64
        random, rows are B (64): sequential 8 reversed 8 pair first 8 one rounding 8
        tie element 0x1.000002p+0, midpoint element 0x1.000002p+0; element (8, 9) 0x1.42cp+4 (sequential 0x1.42cp+4) |}]
