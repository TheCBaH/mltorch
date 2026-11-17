(* A chain of pointwise stages run block by block through view calls on rolling
   windows computes, bit for bit, what the same stages compute over whole
   tensors. A read that would straddle a window's end is refused. *)

open Ssa_ir
module S = Band_schedule
module D = Band_driver

let count = S.Count.of_int

let kernel ?(extent = 1) name ~chunk =
  {
    S.name;
    extent = count extent;
    stride = count 1;
    pad = count 0;
    dilation = count 1;
    chunk = S.Span.chunk chunk;
  }

let stage ?extent name ~chunk ~cin ~cout =
  {
    D.Stage.kernel = kernel ?extent name ~chunk;
    cin = Int64.of_int cin;
    cout = Int64.of_int cout;
  }

let lcg = ref 3

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let elements (p : D.Program.t) id =
  let b = Option.get (Ssa_program.find_buffer p.D.Program.program id) in
  Int64.to_int (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents))

let index_of id l =
  let rec go i = function
    | [] -> None
    | x :: rest -> if Ssa_id.Buffer.equal x id then Some i else go (i + 1) rest
  in
  go 0 l

let run (p : D.Program.t) ~input ~weights =
  let memory =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let id = b.Ssa_buffer.id in
        let cells =
          if Ssa_id.Buffer.equal id p.D.Program.input then input
          else
            match index_of id p.D.Program.weights with
            | Some i -> List.nth weights i
            | None -> Array.make (elements p id) 0.
        in
        Ssa_id.Buffer.Map.add id (Ssa_memory.Floats cells) m)
      Ssa_id.Buffer.Map.empty p.D.Program.program.Ssa_program.buffers
  in
  Err.or_raise ~pp_error:Ssa_interp.pp_error
    (Ssa_interp.run p.D.Program.program ~memory);
  match Ssa_id.Buffer.Map.find p.D.Program.output memory with
  | Ssa_memory.Floats cs -> cs
  | _ -> assert false

let case ?mutation ~name stages ~width ~rows ~block =
  match
    D.program ?mutation stages ~width ~rows:(count rows)
      ~block:(S.Span.block block)
  with
  | Error r -> Fmt.pr "%s: refused: %a@." name D.pp_refusal r
  | Ok banded ->
      let reference = D.whole stages ~width ~rows:(count rows) in
      let input =
        Array.init (elements reference reference.D.Program.input) (fun _ ->
            value ())
      in
      let weights =
        List.map
          (fun id -> Array.init (elements reference id) (fun _ -> value ()))
          reference.D.Program.weights
      in
      let want = run reference ~input ~weights in
      let got = run banded ~input ~weights in
      let unrolled =
        match
          D.program ~looped:false stages ~width ~rows:(count rows)
            ~block:(S.Span.block block)
        with
        | Ok u -> run u ~input ~weights
        | Error _ -> assert false
      in
      let same a b =
        Array.for_all2
          (fun a b -> Int32.bits_of_float a = Int32.bits_of_float b)
          a b
      in
      Fmt.pr "%s: windows %a; loop %a; %s; unrolled %s@." name
        Fmt.(list ~sep:(any " ") int)
        (List.map (fun (_, w) -> S.Count.to_int w) banded.D.Program.rings)
        Fmt.(
          option ~none:(any "none") (fun ppf (l : D.Loop.t) ->
              pf ppf "%d blocks x %d from block %d"
                (S.Count.to_int l.D.Loop.period)
                (S.Count.to_int l.D.Loop.iterations)
                (S.Count.to_int l.D.Loop.first)))
        banded.D.Program.loop
        (if same want got then "equals the whole-tensor Contracts"
         else "DIFFERS")
        (if same want unrolled then "equal" else "DIFFERS")

let%expect_test "banded pointwise chains equal the whole-tensor chain" =
  let three =
    [
      stage "a" ~chunk:2 ~cin:2 ~cout:3;
      stage "b" ~chunk:2 ~cin:3 ~cout:4;
      stage "c" ~chunk:2 ~cin:4 ~cout:2;
    ]
  in
  case ~name:"3 stages, 8 rows, block 2" three ~width:3L ~rows:8 ~block:2;
  case ~name:"3 stages, 7 rows, block 2" three ~width:2L ~rows:7 ~block:2;
  case ~name:"3 stages, 8 rows, block 3" three ~width:3L ~rows:8 ~block:3;
  case ~name:"3 stages, 16 rows, block 2" three ~width:3L ~rows:16 ~block:2;
  case ~name:"3 stages, 23 rows, block 3" three ~width:2L ~rows:23 ~block:3;
  case ~name:"3 stages, 12 rows, block 1" three ~width:2L ~rows:12 ~block:1;
  case ~name:"3 stages, 48 rows, block 3" three ~width:2L ~rows:48 ~block:3;
  case ~name:"1 stage, 5 rows, block 5"
    [ stage "a" ~chunk:1 ~cin:2 ~cout:2 ]
    ~width:2L ~rows:5 ~block:5;
  [%expect
    {|
    3 stages, 8 rows, block 2: windows 2 2; loop 1 blocks x 4 from block 0; equals the whole-tensor Contracts; unrolled equal
    3 stages, 7 rows, block 2: windows 2 2; loop 1 blocks x 3 from block 0; equals the whole-tensor Contracts; unrolled equal
    3 stages, 8 rows, block 3: windows 2 4; loop none; equals the whole-tensor Contracts; unrolled equal
    3 stages, 16 rows, block 2: windows 2 2; loop 1 blocks x 8 from block 0; equals the whole-tensor Contracts; unrolled equal
    3 stages, 23 rows, block 3: windows 2 4; loop none; equals the whole-tensor Contracts; unrolled equal
    3 stages, 12 rows, block 1: windows 2 2; loop 2 blocks x 6 from block 0; equals the whole-tensor Contracts; unrolled equal
    3 stages, 48 rows, block 3: windows 2 4; loop 4 blocks x 4 from block 0; equals the whole-tensor Contracts; unrolled equal
    1 stage, 5 rows, block 5: windows ; loop none; equals the whole-tensor Contracts; unrolled equal |}]

let%expect_test "a kernel with stride or padding is refused" =
  let k = { (kernel "w" ~chunk:1) with S.stride = count 2 } in
  case ~name:"stride"
    [ { D.Stage.kernel = k; cin = 1L; cout = 1L } ]
    ~width:2L ~rows:6 ~block:2;
  [%expect
    {| stride: refused: w is not a stride 1 kernel without padding or dilation (the rows a window reads are not consecutive producer rows) |}]

(* A 3x3 stage reads a ring of producer rows, three at a time. *)
let windowed =
  [
    stage "a" ~chunk:2 ~cin:2 ~cout:3;
    stage ~extent:3 "b" ~chunk:2 ~cin:3 ~cout:2;
  ]

let window_chains =
  let w3 = stage ~extent:3 in
  [
    ("pointwise then 3x3, 10 rows, block 2", windowed, 5L, 10, 2);
    ("pointwise then 3x3, 11 rows, block 3", windowed, 6L, 11, 3);
    ( "3x3 then pointwise, 12 rows, block 2",
      [ w3 "a" ~chunk:2 ~cin:2 ~cout:3; stage "b" ~chunk:2 ~cin:3 ~cout:2 ],
      5L,
      12,
      2 );
    ( "3x3, pointwise, 3x3, 20 rows, block 2",
      [
        w3 "a" ~chunk:2 ~cin:2 ~cout:3;
        stage "b" ~chunk:2 ~cin:3 ~cout:2;
        w3 "c" ~chunk:2 ~cin:2 ~cout:2;
      ],
      6L,
      20,
      2 );
    ( "3x3 then 3x3, 30 rows, block 1",
      [ w3 "a" ~chunk:1 ~cin:2 ~cout:2; w3 "b" ~chunk:1 ~cin:2 ~cout:3 ],
      5L,
      30,
      1 );
    ( "one 3x3 stage, 9 rows, block 4",
      [ w3 "a" ~chunk:1 ~cin:2 ~cout:2 ],
      4L,
      9,
      4 );
  ]

let%expect_test "banded 3x3 chains equal the whole-tensor chain" =
  List.iter
    (fun (name, stages, width, rows, block) ->
      case ~name stages ~width ~rows ~block)
    window_chains;
  [%expect
    {|
    pointwise then 3x3, 10 rows, block 2: windows 4; loop 6 blocks x 2 from block 1; equals the whole-tensor Contracts; unrolled equal
    pointwise then 3x3, 11 rows, block 3: windows 6; loop 3 blocks x 2 from block 1; equals the whole-tensor Contracts; unrolled equal
    3x3 then pointwise, 12 rows, block 2: windows 2; loop 1 blocks x 5 from block 0; equals the whole-tensor Contracts; unrolled equal
    3x3, pointwise, 3x3, 20 rows, block 2: windows 2 4; loop 12 blocks x 3 from block 1; equals the whole-tensor Contracts; unrolled equal
    3x3 then 3x3, 30 rows, block 1: windows 3; loop 9 blocks x 8 from block 1; equals the whole-tensor Contracts; unrolled equal
    one 3x3 stage, 9 rows, block 4: windows ; loop 3 blocks x 7 from block 0; equals the whole-tensor Contracts; unrolled equal |}]

(* Reading the window's first row for every row it covers is seen. *)
let%expect_test "a dropped window offset is red" =
  List.iter
    (fun (name, stages, width, rows, block) ->
      case ~mutation:D.Window_offset_dropped ~name stages ~width ~rows ~block)
    window_chains;
  [%expect
    {|
    pointwise then 3x3, 10 rows, block 2: windows 4; loop 6 blocks x 2 from block 1; DIFFERS; unrolled equal
    pointwise then 3x3, 11 rows, block 3: windows 6; loop 3 blocks x 2 from block 1; DIFFERS; unrolled equal
    3x3 then pointwise, 12 rows, block 2: windows 2; loop 1 blocks x 5 from block 0; DIFFERS; unrolled equal
    3x3, pointwise, 3x3, 20 rows, block 2: windows 2 4; loop 12 blocks x 3 from block 1; DIFFERS; unrolled equal
    3x3 then 3x3, 30 rows, block 1: windows 3; loop 9 blocks x 8 from block 1; DIFFERS; unrolled equal
    one 3x3 stage, 9 rows, block 4: windows ; loop 3 blocks x 7 from block 0; DIFFERS; unrolled equal |}]

(* Where a target has no form for a view call it refuses, by name; the
   expansion is how a program with one reaches such a target. *)
let%expect_test
    "C, JavaScript and Wasm refuse a view call; Machine IR lowers it" =
  let stages =
    [ stage "a" ~chunk:2 ~cin:2 ~cout:3; stage "b" ~chunk:2 ~cin:3 ~cout:2 ]
  in
  let p =
    match
      D.program stages ~width:2L ~rows:(count 4) ~block:(S.Span.block 2)
    with
    | Ok p -> p.D.Program.program
    | Error _ -> assert false
  in
  let c = function
    | Ok _ -> "accepted"
    | Error e -> Fmt.str "%a" Ssa_c.pp_error e
  in
  let js = function
    | Ok _ -> "accepted"
    | Error e -> Fmt.str "%a" Ssa_js.pp_error e
  in
  let wasm = function
    | Ok _ -> "accepted"
    | Error e -> Fmt.str "%a" Ssa_wasm.pp_error e
  in
  let mir = function
    | Ok _ -> "accepted"
    | Error e -> Fmt.str "%a" Machine_lower.Mir_lower.Refusal.pp e
  in
  Fmt.pr "C: %s@." (c (Ssa_c.kernel ~name:"k" p));
  Fmt.pr "JavaScript: %s@." (js (Ssa_js.program p));
  Fmt.pr "Wasm: %s@." (wasm (Ssa_wasm.lower p));
  Fmt.pr "Machine IR: %s@."
    (mir
       (let resolved =
          Ssa_plan.resolve ~numerics:Ssa_numerics.Reference_f64 p
        in
        Err.payload
          (Machine_lower.Mir_lower.program
             ~planning:(Some (Machine_lower.Mir_lower.summary resolved))
             resolved.Ssa_plan.program)));
  let expanded =
    Ssa_precision.to_f32 ~exact:true (Ssa_expand_view_call.program p)
  in
  Fmt.pr "C, expanded: %s@." (c (Ssa_c.kernel ~name:"k" expanded));
  [%expect
    {|
    C: C has no form for a view call
    JavaScript: a view call has no JavaScript form
    Wasm: a view call has no WebAssembly form
    Machine IR: accepted
    C, expanded: accepted |}]

(* The loop's two defects are seen: bases that do not move repeat the first
   repetition's rows, a loop one repetition short leaves the last block's rows
   unwritten. *)
let%expect_test "loop mutations are red" =
  let three =
    [
      stage "a" ~chunk:2 ~cin:2 ~cout:3;
      stage "b" ~chunk:2 ~cin:3 ~cout:4;
      stage "c" ~chunk:2 ~cin:4 ~cout:2;
    ]
  in
  List.iter
    (fun (name, mutation) ->
      Fmt.pr "== %s@." name;
      case ~mutation ~name:"3 stages, 16 rows, block 2" three ~width:3L ~rows:16
        ~block:2)
    D.[ ("iterations short", Iterations_short); ("step dropped", Step_dropped) ];
  [%expect
    {|
    == iterations short
    3 stages, 16 rows, block 2: windows 2 2; loop 1 blocks x 8 from block 0; DIFFERS; unrolled equal
    == step dropped
    3 stages, 16 rows, block 2: windows 2 2; loop 1 blocks x 8 from block 0; DIFFERS; unrolled equal |}]
