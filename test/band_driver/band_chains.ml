(* The driver chains of the band-driver and view-call suites as data, and the
   comparison a native route makes with them: the banded program runs on the
   route, the whole-tensor chain on the SSA interpreter, output bits equal. *)

open Ssa_ir
module S = Band_schedule
module D = Band_driver

let i64 = Int64.of_int
let lcg = ref 5

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let bits x = Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xFFFF_FFFFL

(* The binary32 bit patterns of a program's output on the SSA interpreter. *)
let ssa_output (p : Ssa_program.t) ~inputs =
  let memory =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let cells =
          match
            List.assoc_opt (Ssa_id.Buffer.to_int b.Ssa_buffer.id) inputs
          with
          | Some c -> c
          | None ->
              Array.make
                (Int64.to_int
                   (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents)))
                0.
        in
        Ssa_id.Buffer.Map.add b.Ssa_buffer.id (Ssa_memory.Floats cells) m)
      Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
  in
  Err.or_raise ~pp_error:Ssa_interp.pp_error (Ssa_interp.run p ~memory);
  List.filter_map
    (fun (b : Ssa_buffer.t) ->
      match
        (b.Ssa_buffer.role, Ssa_id.Buffer.Map.find b.Ssa_buffer.id memory)
      with
      | Ssa_buffer.Output, Ssa_memory.Floats cs -> Some (Array.map bits cs)
      | _ -> None)
    p.Ssa_program.buffers

(* The outputs of a successful observation, as binary32 bit patterns. *)
let observed (o : Machine_ir.Mir_observation.t) =
  match o.Machine_ir.Mir_observation.status with
  | Machine_ir.Mir_observation.Status.Success ->
      Ok
        (List.map
           (fun (out : Machine_ir.Mir_observation.Output.t) ->
             Array.map
               (function
                 | Some (c : Machine_ir.Mir_const.t) ->
                     Int64.logand c.Machine_ir.Mir_const.bits 0xFFFF_FFFFL
                 | None -> -1L)
               out.Machine_ir.Mir_observation.Output.cells)
           o.Machine_ir.Mir_observation.outputs)
  | _ -> Error "the run did not succeed"

let kernel ?(extent = 1) name ~chunk =
  {
    S.name;
    extent = S.Count.of_int extent;
    stride = S.Count.of_int 1;
    pad = S.Count.of_int 0;
    dilation = S.Count.of_int 1;
    chunk = S.Span.chunk chunk;
  }

let stage ?extent name ~chunk ~cin ~cout =
  {
    D.Stage.kernel = kernel ?extent name ~chunk;
    cin = i64 cin;
    cout = i64 cout;
  }

type chain = {
  name : string;
  stages : D.Stage.t list;
  width : int64;
  rows : int;
  block : int;
  looped : bool;
}

let three =
  [
    stage "a" ~chunk:2 ~cin:2 ~cout:3;
    stage "b" ~chunk:2 ~cin:3 ~cout:4;
    stage "c" ~chunk:2 ~cin:4 ~cout:2;
  ]

let w3 = stage ~extent:3

let point_then_window =
  [ stage "a" ~chunk:2 ~cin:2 ~cout:3; w3 "b" ~chunk:2 ~cin:3 ~cout:2 ]

(* Pointwise and 3x3 chains, unrolled and looped. *)
let chains =
  [
    {
      name = "pointwise 3 stages, 8 rows";
      stages = three;
      width = 3L;
      rows = 8;
      block = 2;
      looped = false;
    };
    {
      name = "pointwise 3 stages, 16 rows";
      stages = three;
      width = 3L;
      rows = 16;
      block = 2;
      looped = true;
    };
    {
      name = "pointwise 3 stages, 12 rows, block 1";
      stages = three;
      width = 2L;
      rows = 12;
      block = 1;
      looped = true;
    };
    {
      name = "pointwise then 3x3, 10 rows";
      stages = point_then_window;
      width = 5L;
      rows = 10;
      block = 2;
      looped = false;
    };
    {
      name = "pointwise then 3x3, 10 rows";
      stages = point_then_window;
      width = 5L;
      rows = 10;
      block = 2;
      looped = true;
    };
    {
      name = "3x3, pointwise, 3x3, 20 rows";
      stages =
        [
          w3 "a" ~chunk:2 ~cin:2 ~cout:3;
          stage "b" ~chunk:2 ~cin:3 ~cout:2;
          w3 "c" ~chunk:2 ~cin:2 ~cout:2;
        ];
      width = 6L;
      rows = 20;
      block = 2;
      looped = true;
    };
  ]

let elements (p : D.Program.t) id =
  Int64.to_int
    (Option.get
       (Ssa_buffer.elements
          (Option.get (Ssa_program.find_buffer p.D.Program.program id))
            .Ssa_buffer.extents))

(* [run] makes a route's output bits for the banded program over [inputs]; the
   verdict compares them with the whole chain's. *)
let verdict ?driver ~run c =
  match
    D.program ~looped:c.looped ?mutation:driver c.stages ~width:c.width
      ~rows:(S.Count.of_int c.rows) ~block:(S.Span.block c.block)
  with
  | Error _ -> Fmt.str "%s: driver refused" c.name
  | Ok banded ->
      let whole =
        D.whole c.stages ~width:c.width ~rows:(S.Count.of_int c.rows)
      in
      let inputs =
        List.map
          (fun id ->
            ( Ssa_id.Buffer.to_int id,
              Array.init (elements whole id) (fun _ -> value ()) ))
          (whole.D.Program.input :: whole.D.Program.weights)
      in
      let want = ssa_output whole.D.Program.program ~inputs in
      Fmt.str "%s, %s: %s" c.name
        (if banded.D.Program.loop = None then "unrolled" else "looped")
        (match run banded.D.Program.program ~inputs with
        | Error e -> e
        | Ok got -> if got = want then "equal" else "DIFFERS")
