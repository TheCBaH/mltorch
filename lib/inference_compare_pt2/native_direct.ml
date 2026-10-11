(* The ordinary graph runner over the Native direct evaluator for any Native
   graph model: the lowered graph, the canonical one, or a copy of either
   described under another key. The engine is the one {!Native_runner} names;
   the graph under it is the model's, not an archive's. *)

module Tid = Graph_ir.Tensor_id

module Dialect = Graph_dialect.Make (struct
  type op = Graph_ir.op

  let ops = Graph_model.native_ops
end)

module Runner (D : sig
  val dots : Direct.dot_accumulation
end) =
Graph_runner.Make (struct
  type op = Graph_ir.op
  type compiled = unit

  let compile _ = Ok ()

  let execute () (g : Graph_ir.graph) ~inputs =
    match
      Direct.with_dot_accumulation D.dots (fun () -> Eval_direct.run g ~inputs)
    with
    | Error e ->
        Error
          (Format.asprintf "Native: %a" Eval_direct.pp_error (Err.Error.kind e))
    | Ok env ->
        let rec outs acc = function
          | [] -> Ok (List.rev acc)
          | id :: rest -> (
              match Tid.Map.find_opt id env with
              | None -> Error "a declared output was not produced"
              | Some packed -> outs ((id, packed) :: acc) rest)
        in
        Result.map (List.map snd) (outs [] g.outputs)

  let release () = ()
end)

module Binary64 = Runner (struct
  let dots = Direct.Binary64
end)

module Sequential32 = Runner (struct
  let dots = Direct.Binary32_sequential
end)

let identity = Native_runner.identity

let create ?artifact ?(dots = Direct.Binary64) () :
    Graph_ir.op Graph_body.t Graph_runner.t =
  let identity () = identity ?artifact ~dots () in
  match dots with
  | Direct.Binary64 -> Binary64.create ~identity
  | Direct.Binary32_sequential -> Sequential32.create ~identity
