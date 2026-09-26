(* See loop_node_program.mli. *)

type error = [ `Adapt of Kernel_adapt.error | `Lower of Loop_lower.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Adapt e -> Kernel_adapt.pp_error fmt e
  | `Lower e -> Loop_lower.pp_error fmt e

let lower ?(limits = Kernel.Limits.default) (g : Graph_ir.graph)
    (node : Graph_ir.node) ~(output : Output_ordinal.t) :
    (Loop_program.t, error) Err.t =
  let open Err.Syntax in
  let oid =
    List.assoc output (Output_ordinal.indexed node.Graph_ir.Node.outputs)
  in
  let program = Eval_symbolic.node_program ~limits g node in
  (* [~select] must narrow to just [oid] too, not only [~outputs]: with
     [~select] absent (whole-program), [Kernel_adapt.required]'s own
     [graph_outs] step includes EVERY entry of [Stage_program.outputs] that
     falls in the selection -- for a multi-output node (`Unbind`, `Lstm`,
     `Max_dim`'s two outputs, ...) that is every sibling output, not just
     [oid], so a single-element [~outputs] then fails to "begin with" the
     multi-element required list. Narrowing [~select] to [oid] alone makes
     the required list exactly [[oid]], matching [~outputs] again. *)
  let* kernel =
    Kernel_adapt.of_stage_program ~limits
      ~select:(Tensor_id.Set.singleton oid)
      ~outputs:[ oid ] program
    |> Err.map_error (fun e -> `Adapt e)
  in
  Loop_lower.lower (Fusion_plan.default kernel)
  |> Err.map_error (fun e -> `Lower e)
