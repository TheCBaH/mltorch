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

let axis ?(extent = 1) ?(stride = 1) ?(pad = 0) ?(dilation = 1) () =
  {
    S.Axis.extent = S.Count.of_int extent;
    stride = S.Count.of_int stride;
    pad = S.Count.of_int pad;
    dilation = S.Count.of_int dilation;
  }

(* A stage whose row and column axes are given apart. *)
let shaped name ~chunk ~cin ~cout ~rows ~cols =
  {
    D.Stage.kernel = { S.name; rows; cols; chunk = S.Span.chunk chunk };
    cin = i64 cin;
    cout = i64 cout;
  }

let stage ?extent name ~chunk ~cin ~cout =
  let a = axis ?extent () in
  shaped name ~chunk ~cin ~cout ~rows:a ~cols:a

(* The chain's output by nested sums, independent of the driver and of the
   Contract chain: per output position the window's rows, then columns, then
   channels, each product rounded to binary32 and added; a position in the
   padding contributes nothing. The input is rows x width x cin, the weights of
   a stage (kh * kw * cin) x cout. *)
let oracle stages ~width ~rows ~input ~weights =
  let cnt = S.Count.to_int in
  let step (h, w, cur) ((st : D.Stage.t), wt) =
    let k = st.D.Stage.kernel in
    let ra = k.S.rows and ca = k.S.cols in
    let cin = Int64.to_int st.D.Stage.cin
    and cout = Int64.to_int st.D.Stage.cout in
    let kh = cnt ra.S.Axis.extent and kw = cnt ca.S.Axis.extent in
    let ho = cnt (S.output_height k ~input:(S.Count.of_int h)) in
    let wo =
      (w + (2 * cnt ca.S.Axis.pad) - cnt (S.Axis.window ca))
      / cnt ca.S.Axis.stride
      + 1
    in
    let out = Array.make (ho * wo * cout) 0. in
    for y = 0 to ho - 1 do
      for x = 0 to wo - 1 do
        for o = 0 to cout - 1 do
          let acc = ref 0. in
          for ky = 0 to kh - 1 do
            for kx = 0 to kw - 1 do
              let iy =
                (y * cnt ra.S.Axis.stride)
                - cnt ra.S.Axis.pad
                + (ky * cnt ra.S.Axis.dilation)
              and ix =
                (x * cnt ca.S.Axis.stride)
                - cnt ca.S.Axis.pad
                + (kx * cnt ca.S.Axis.dilation)
              in
              if iy >= 0 && iy < h && ix >= 0 && ix < w then
                for c = 0 to cin - 1 do
                  let a = cur.((((iy * w) + ix) * cin) + c)
                  and b = wt.((((((ky * kw) + kx) * cin) + c) * cout) + o) in
                  acc := F32_math.round32 (!acc +. F32_math.round32 (a *. b))
                done
            done
          done;
          out.((((y * wo) + x) * cout) + o) <- !acc
        done
      done
    done;
    (ho, wo, out)
  in
  let _, _, out =
    List.fold_left step
      (rows, Int64.to_int width, input)
      (List.combine stages weights)
  in
  [ Array.map bits out ]

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

(* Pointwise and square-window chains, unrolled and looped. *)
let square_chains =
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

(* Windows that are not square, and strides, paddings and dilations that differ
   by axis. A channel pair per stage keeps the products small. *)
let shaped_chains =
  let a = axis in
  let w name ~rows ~cols ~cin ~cout =
    shaped name ~chunk:2 ~cin ~cout ~rows ~cols
  in
  let pw name ~cin ~cout = stage name ~chunk:2 ~cin ~cout in
  [
    {
      name = "pointwise then 1x3 pad 1 along the columns, 14 rows";
      stages =
        [
          pw "a" ~cin:2 ~cout:3;
          w "b" ~rows:(a ()) ~cols:(a ~extent:3 ~pad:1 ()) ~cin:3 ~cout:2;
        ];
      width = 6L;
      rows = 14;
      block = 2;
      looped = true;
    };
    {
      name = "pointwise then 3x1 pad 1 along the rows, 14 rows";
      stages =
        [
          pw "a" ~cin:2 ~cout:3;
          w "b" ~rows:(a ~extent:3 ~pad:1 ()) ~cols:(a ()) ~cin:3 ~cout:2;
        ];
      width = 6L;
      rows = 14;
      block = 2;
      looped = true;
    };
    {
      name = "3x5 pad 1 by 2 then pointwise, 12 rows";
      stages =
        [
          w "a" ~rows:(a ~extent:3 ~pad:1 ()) ~cols:(a ~extent:5 ~pad:2 ())
            ~cin:2 ~cout:3;
          pw "b" ~cin:3 ~cout:2;
        ];
      width = 9L;
      rows = 12;
      block = 2;
      looped = true;
    };
    {
      name = "3x3 stride 2 by 1 then 5x3 stride 1 by 2 pad 1, 21 rows";
      stages =
        [
          w "a" ~rows:(a ~extent:3 ~stride:2 ()) ~cols:(a ~extent:3 ()) ~cin:2
            ~cout:2;
          w "b" ~rows:(a ~extent:5 ~pad:2 ())
            ~cols:(a ~extent:3 ~stride:2 ~pad:1 ())
            ~cin:2 ~cout:3;
        ];
      width = 11L;
      rows = 21;
      block = 2;
      looped = true;
    };
    {
      name = "3x3 dilation 2 by 1, pad 2 by 0, 1x1 stride 1 by 2, 20 rows";
      stages =
        [
          w "a"
            ~rows:(a ~extent:3 ~pad:2 ~dilation:2 ())
            ~cols:(a ~extent:3 ()) ~cin:2 ~cout:2;
          w "b" ~rows:(a ()) ~cols:(a ~stride:2 ()) ~cin:2 ~cout:2;
        ];
      width = 7L;
      rows = 20;
      block = 2;
      looped = true;
    };
  ]

let chains = square_chains @ shaped_chains

let elements (p : D.Program.t) id =
  Int64.to_int
    (Option.get
       (Ssa_buffer.elements
          (Option.get (Ssa_program.find_buffer p.D.Program.program id))
            .Ssa_buffer.extents))

(* {!oracle} over a program's [inputs] (buffer id to cells), the ids those of
   [whole]. *)
let oracle_of_inputs stages ~width ~rows ~(whole : D.Program.t) ~inputs =
  let cells id = List.assoc (Ssa_id.Buffer.to_int id) inputs in
  oracle stages ~width ~rows
    ~input:(cells whole.D.Program.input)
    ~weights:(List.map cells whole.D.Program.weights)

(* [run] makes a route's output bits for the banded program over [inputs]; the
   verdict compares them with the nested sums over the same inputs (the whole
   chain must equal them too). *)
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
      let want =
        oracle_of_inputs c.stages ~width:c.width ~rows:c.rows ~whole ~inputs
      in
      Fmt.str "%s, %s: %s" c.name
        (if banded.D.Program.loop = None then "unrolled" else "looped")
        (if ssa_output whole.D.Program.program ~inputs <> want then
           "the whole chain differs from the nested sums"
         else
           match run banded.D.Program.program ~inputs with
           | Error e -> e
           | Ok got -> if got = want then "equal" else "DIFFERS")
