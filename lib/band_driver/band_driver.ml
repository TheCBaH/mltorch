open Ssa_ir
module B = Ssa_builder
module S = Band_schedule

module Stage = struct
  type t = { kernel : S.kernel; cin : int64; cout : int64 }
end

type refusal =
  | Not_pointwise of string
  | Read_spans_wrap of { kernel : string; lo : S.Row.t; hi : S.Row.t }

let pp_refusal ppf = function
  | Not_pointwise k ->
      Fmt.pf ppf "%s is not a pointwise kernel (extent 1, stride 1, no padding)"
        k
  | Read_spans_wrap { kernel; lo; hi } ->
      Fmt.pf ppf
        "%s reads rows [%a, %a) across the end of its producer's window" kernel
        S.Row.pp lo S.Row.pp hi

module Program = struct
  type t = {
    program : Ssa_program.t;
    input : Ssa_id.Buffer.t;
    weights : Ssa_id.Buffer.t list;
    output : Ssa_id.Buffer.t;
    rings : (Ssa_id.Buffer.t * S.Count.t) list;
  }
end

let id = Ssa_id.Buffer.of_int
let i64 = Int64.of_int

let pointwise (k : S.kernel) =
  S.Count.to_int k.S.extent = 1
  && S.Count.to_int k.S.stride = 1
  && S.Count.to_int k.S.pad = 0
  && S.Count.to_int k.S.dilation = 1

(* The product of one stage over [pixels] pixels: A is pixels x cin, B the
   weights cin x cout, C pixels x cout. *)
let desc (st : Stage.t) ~pixels =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:1L ~m:pixels ~n:st.Stage.cout ~k:st.Stage.cin
          ~a:{ Contraction.Operand.batch = 0L; outer = st.Stage.cin; k = 1L }
          ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = st.Stage.cout }
          ~c:{ Contraction.Output.batch = 0L; row = st.Stage.cout; col = 1L }
          ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero
          ~numerics:Contraction.Numerics.Ordered))

let buffer n ~pixels ~channels role =
  {
    Ssa_buffer.id = id n;
    extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:pixels ~w:channels ~c:1L;
    format = Ssa_format.F32;
    role;
  }

(* The buffers of a chain whose stage [i]'s output has [out_rows.(i)] rows: the
   input is 0, the weights 1 to n, the intermediates n + 1 to 2n - 1 and the
   output 2n. *)
let layout stages ~width ~rows ~out_rows =
  let n = List.length stages in
  let stages = Array.of_list stages in
  let first = stages.(0) in
  let input =
    buffer 0
      ~pixels:(Int64.mul (i64 rows) width)
      ~channels:first.Stage.cin Ssa_buffer.Input
  in
  let weights =
    List.init n (fun i ->
        buffer (1 + i) ~pixels:stages.(i).Stage.cin
          ~channels:stages.(i).Stage.cout Ssa_buffer.Input)
  in
  let outs =
    List.init n (fun i ->
        buffer
          (if i = n - 1 then 2 * n else n + 1 + i)
          ~pixels:(Int64.mul (i64 out_rows.(i)) width)
          ~channels:stages.(i).Stage.cout
          (if i = n - 1 then Ssa_buffer.Output else Ssa_buffer.Scratch))
  in
  (input :: weights) @ outs

let ids bufs = List.map (fun (b : Ssa_buffer.t) -> b.Ssa_buffer.id) bufs

let whole stages ~width ~rows =
  let n = List.length stages in
  let rows = S.Count.to_int rows in
  let out_rows = Array.make n rows in
  let bufs = layout stages ~width ~rows ~out_rows in
  let all = ids bufs in
  let input = List.hd all in
  let weights = List.filteri (fun i _ -> i >= 1 && i <= n) all in
  let outs = List.filteri (fun i _ -> i > n) all in
  let built =
    Err.or_raise ~pp_error:Ssa_verify.pp_error
      (B.program ~buffers:bufs (fun bld ->
           let pixels = Int64.mul (i64 rows) width in
           List.iteri
             (fun i (st : Stage.t) ->
               let src = if i = 0 then input else List.nth outs (i - 1) in
               B.contract bld (desc st ~pixels) ~a:src ~b:(List.nth weights i)
                 ~c:(List.nth outs i))
             stages))
  in
  {
    Program.program = built;
    input;
    weights;
    output = List.nth outs (n - 1);
    rings = [];
  }

let program stages ~width ~rows ~block =
  match
    List.find_opt (fun (s : Stage.t) -> not (pointwise s.Stage.kernel)) stages
  with
  | Some s -> Error (Not_pointwise s.Stage.kernel.S.name)
  | None -> (
      let n = List.length stages in
      let kernels = List.map (fun (s : Stage.t) -> s.Stage.kernel) stages in
      let schedule = S.run kernels ~input:rows ~block in
      let rows_i = S.Count.to_int rows in
      let windows =
        Array.of_list
          (List.mapi
             (fun i w ->
               if i = n - 1 then None
               else
                 Some
                   (S.Window.make ~rows:w ~chunk:(List.nth kernels i).S.chunk))
             schedule.S.windows)
      in
      let out_rows =
        Array.map
          (function
            | Some w -> S.Count.to_int (S.Window.rows w) | None -> rows_i)
          windows
      in
      let bufs = layout stages ~width ~rows:rows_i ~out_rows in
      let all = ids bufs in
      let input = List.hd all in
      let weights = List.filteri (fun i _ -> i >= 1 && i <= n) all in
      let outs = List.filteri (fun i _ -> i > n) all in
      let stage_index name =
        let rec find i = function
          | [] -> invalid_arg "Band_driver: an event of no stage"
          | (s : Stage.t) :: rest ->
              if String.equal s.Stage.kernel.S.name name then i
              else find (i + 1) rest
        in
        find 0 stages
      in
      let stages_a = Array.of_list stages in
      let spans =
        List.find_map
          (fun (e : S.event) ->
            let i = stage_index e.S.kernel in
            if i = 0 then None
            else
              match windows.(i - 1) with
              | Some w ->
                  let lo, hi = e.S.consumed in
                  if S.Window.spans_wrap w ~lo ~hi then
                    Some (Read_spans_wrap { kernel = e.S.kernel; lo; hi })
                  else None
              | None -> None)
          schedule.S.events
      in
      match spans with
      | Some r -> Error r
      | None ->
          let built =
            Err.or_raise ~pp_error:Ssa_verify.pp_error
              (B.program ~buffers:bufs (fun bld ->
                   List.iter
                     (fun (e : S.event) ->
                       let i = stage_index e.S.kernel in
                       let st = stages_a.(i) in
                       let plo, phi = e.S.produced in
                       let clo, _ = e.S.consumed in
                       let pixels =
                         Int64.mul
                           (i64 (S.Row.to_int phi - S.Row.to_int plo))
                           width
                       in
                       let read_row =
                         if i = 0 then S.Row.to_int clo
                         else
                           match windows.(i - 1) with
                           | Some w ->
                               S.Ring_row.to_int (S.Window.ring_row w clo)
                           | None -> assert false
                       in
                       let write_row =
                         match windows.(i) with
                         | Some w -> S.Ring_row.to_int (S.Window.ring_row w plo)
                         | None -> S.Row.to_int plo
                       in
                       let base row channels =
                         B.index bld
                           (Int64.mul (Int64.mul (i64 row) width) channels)
                       in
                       B.view_call bld (desc st ~pixels)
                         ~a:
                           ( (if i = 0 then input else List.nth outs (i - 1)),
                             base read_row st.Stage.cin )
                         ~b:(List.nth weights i, B.index bld 0L)
                         ~c:(List.nth outs i, base write_row st.Stage.cout))
                     schedule.S.events))
          in
          Ok
            {
              Program.program = built;
              input;
              weights;
              output = List.nth outs (n - 1);
              rings =
                List.filter_map
                  (fun i ->
                    Option.map
                      (fun w -> (List.nth outs i, S.Window.rows w))
                      windows.(i))
                  (List.init n Fun.id);
            })
