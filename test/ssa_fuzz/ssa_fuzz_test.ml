open Fuzz

let size = default_size

let fail_on = function
  | Ok _ -> ()
  | Error d -> Fmt.failwith "%a" pp_disagreement d

(* The same seeds build the same programs on every runtime: a digest of the
   printed programs of the first seeds is committed here, and js_of_ocaml (32-bit
   int, its own float printing) must produce the same one as native code. *)
let%expect_test "the programs of the first seeds are the same on every runtime"
    =
  let b = Buffer.create 4096 in
  for s = 0 to 199 do
    let case = generate ~seed:(Int64.of_int s) ~size () in
    Buffer.add_string b (Ssa_ir.Ssa_pp.to_string case.program);
    Array.iter
      (fun x -> Buffer.add_string b (Printf.sprintf "%h;" x))
      case.input
  done;
  Fmt.pr "%s@." (Digest.to_hex (Digest.string (Buffer.contents b)));
  [%expect {| 5b9ce5a1eb6e471acd34b663bad84d0b |}]

let%expect_test "every seed builds a program that verifies and replays" =
  let again = ref true in
  for s = 0 to seeds - 1 do
    let seed = Int64.of_int s in
    let a = generate ~seed ~size () and b = generate ~seed ~size () in
    if
      Ssa_ir.Ssa_pp.to_string a.program <> Ssa_ir.Ssa_pp.to_string b.program
      || not (Array.for_all2 Core.Float_bits.equal_portable a.input b.input)
    then again := false
  done;
  Fmt.pr "%d seeds, same seed same program: %b@." seeds !again;
  [%expect {| 2000 seeds, same seed same program: true |}]

let%expect_test "the interpreters agree on every seed, and the rows are reached"
    =
  let reached = Hashtbl.create 16 and ok = ref 0 in
  for s = 0 to seeds - 1 do
    let case = generate ~seed:(Int64.of_int s) ~size () in
    match check case [ cfg ] with
    | Error d -> fail_on (Error d)
    | Ok o -> (
        match o.row with
        | None -> incr ok
        | Some e ->
            let n = row_name e in
            Hashtbl.replace reached n
              (1 + Option.value ~default:0 (Hashtbl.find_opt reached n)))
  done;
  let targeted =
    [
      "coord_out_of_range";
      "gather_index_out_of_range";
      "i64_division_by_zero";
      "i64_division_overflow";
      "i64_from_float_infinite";
      "i64_from_float_nan";
      "i64_from_float_out_of_range";
      "index_overflow";
      "scan_meter";
      "scan_projection";
      "unbound_local";
    ]
  in
  List.iter
    (fun n -> Fmt.pr "%s reached: %b@." n (Hashtbl.mem reached n))
    targeted;
  (* Cells are only compared on a success, so a generator whose programs all
     fail early proves nothing about them. *)
  Fmt.pr "a quarter of the programs run to the end: %b@." (!ok * 4 >= seeds);
  [%expect
    {|
    coord_out_of_range reached: true
    gather_index_out_of_range reached: true
    i64_division_by_zero reached: true
    i64_division_overflow reached: true
    i64_from_float_infinite reached: true
    i64_from_float_nan reached: true
    i64_from_float_out_of_range reached: true
    index_overflow reached: true
    scan_meter reached: true
    scan_projection reached: true
    unbound_local reached: true
    a quarter of the programs run to the end: true |}]

(* ---- the harness reports a defect ----------------------------------------- *)

(* Backends with a defect seeded on the outcome, as a refactor of an emitter
   could introduce one. Each must make the differential fail, and the failure
   must name the seed, the size, the program and the rows. *)
let mutant name f =
  {
    name;
    run = (fun case -> match interp.run case with Ran o -> Ran (f o) | r -> r);
  }

let swallow_failures =
  mutant "swallow" (fun o ->
      match o.row with Some _ -> { row = None; cells = [] } | None -> o)

let confuse_division =
  mutant "confuse" (fun o ->
      match o.row with
      | Some `I64_division_by_zero ->
          { o with row = Some `I64_division_overflow }
      | _ -> o)

let nudge_cell =
  mutant "nudge" (fun o ->
      match (o.row, o.cells) with
      | None, x :: rest -> { o with cells = (x +. 1.) :: rest }
      | _ -> o)

let first_caught backend =
  let rec go s =
    if s >= seeds then None
    else
      match check (generate ~seed:(Int64.of_int s) ~size ()) [ backend ] with
      | Error d -> Some d
      | Ok _ -> go (s + 1)
  in
  go 0

let%expect_test "a seeded defect is caught by some seed" =
  List.iter
    (fun b -> Fmt.pr "%s caught: %b@." b.name (Option.is_some (first_caught b)))
    [ swallow_failures; confuse_division; nudge_cell ];
  [%expect
    {|
    swallow caught: true
    confuse caught: true
    nudge caught: true |}]

let%expect_test
    "a disagreement prints the seed, the size, the program and the rows, at \
     the smallest size that still shows it" =
  let rec first s =
    match minimize ~seed:(Int64.of_int s) ~size:default_size [ nudge_cell ] with
    | Some d -> d
    | None -> first (s + 1)
  in
  Fmt.pr "%a@." pp_disagreement (first 0);
  [%expect
    {|
    nudge disagrees with the interpreter at seed=0 size=1
    buffer b0 in f32 [1, 1, 1, 1, 8, 1]
    buffer b1 out f32 [1, 1, 1, 1, 8, 1]
    entry(%0:effect) {
      (%1:index) = const 7:index
      (%2:i64) = const 1258811053029615386:i64
      (%3:f32) = convert.i64_to_f32 %2
      (%4:f64) = convert.f32_to_f64 %3
      (%5:index) = const 0:index
      (%6:index) = const 0:index
      (%7:effect) = store.f32_round b1[%6, %6, %6, %5, %1, %6], %4 effect %0
      (%8:index) = const 6:index
      (%9:index) = const 0:index
      (%10:pred) = index.compare.lt %8, %9
      (%11:effect) = if %10 {
        (%12:index) = const 3:index
        (%13:index) = const 6:index
        (%14:index) = const 7:index
        (%15:index) = index.clamp_low %13
        (%16:index) = index.min %15, %14
        (%17:index) = const 0:index
        (%18:index) = const 0:index
        (%19:f64, %20:effect) = load.f32_to_f64 b0[%18, %18, %18, %17, %16, %18] effect %7
        (%21:index) = const 0:index
        (%22:index) = const 0:index
        (%23:effect) = store.f32_round b1[%22, %22, %22, %21, %12, %22], %19 effect %20
        yield %23
      } else {
        yield %7
      }
      yield %11
    }

    interp: ok: 0 0 0 0 0 0 0 1.25881e+18
    nudge: ok: 1 0 0 0 0 0 0 1.25881e+18 |}]

(* A generator that never emits a construct proves nothing about it: each family
   must appear in the printed programs of the seed range, and the programs must
   nest regions. *)
let%expect_test "the generator emits control flow, memory checks and the meter"
    =
  let contains text sub =
    let n = String.length sub and m = String.length text in
    let rec go i = i + n <= m && (String.sub text i n = sub || go (i + 1)) in
    go 0
  in
  let families =
    [
      ("for", ") = for ");
      ("if", ") = if ");
      ("ordered_sum", "ordered_sum ");
      ("local", "local.alloc");
      ("check_local", "check_local");
      ("check_scan", "check_scan");
      ("check_gather", "check_gather");
      ("meter", "meter.");
      ("nested region", "\n      (");
    ]
  in
  let counts = Hashtbl.create 16 in
  for s = 0 to seeds - 1 do
    let text =
      Ssa_ir.Ssa_pp.to_string (generate ~seed:(Int64.of_int s) ~size ()).program
    in
    List.iter
      (fun (name, sub) ->
        if contains text sub then
          Hashtbl.replace counts name
            (1 + Option.value ~default:0 (Hashtbl.find_opt counts name)))
      families
  done;
  List.iter
    (fun (name, _) ->
      (* a region inside a region is rarer than any one construct *)
      let share = if name = "nested region" then 50 else 10 in
      Fmt.pr "%s in a 1/%d of the programs: %b@." name share
        (Option.value ~default:0 (Hashtbl.find_opt counts name) * share >= seeds))
    families;
  [%expect
    {|
    for in a 1/10 of the programs: true
    if in a 1/10 of the programs: true
    ordered_sum in a 1/10 of the programs: true
    local in a 1/10 of the programs: true
    check_local in a 1/10 of the programs: true
    check_scan in a 1/10 of the programs: true
    check_gather in a 1/10 of the programs: true
    meter in a 1/10 of the programs: true
    nested region in a 1/50 of the programs: true |}]
