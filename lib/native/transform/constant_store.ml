open Graph_ir

type error =
  [ `Binding_overwrite of Tensor_id.t
  | Const_ssa.error
  | `Missing_export of Const_ssa.Value_id.t ]

type t = {
  materialized : Tensor.packed Tensor_id.Map.t;
  plan : Const_ssa.t;
  bindings : Const_ssa.Value_id.t Tensor_id.Map.t;
}

let empty =
  {
    materialized = Tensor_id.Map.empty;
    plan = Const_ssa.empty;
    bindings = Tensor_id.Map.empty;
  }

let materialized t = t.materialized
let plan t = t.plan
let binding t id = Tensor_id.Map.find_opt id t.bindings
let bindings t = Tensor_id.Map.bindings t.bindings

let is_effective_constant t id =
  Tensor_id.Map.mem id t.materialized || Option.is_some (binding t id)

let pp_error ppf : [< error ] -> unit = function
  | `Binding_overwrite id ->
      Fmt.pf ppf "constant binding for %a already exists" Tensor_id.pp id
  | #Const_ssa.error as e -> Const_ssa.pp_error ppf e
  | `Missing_export value ->
      Fmt.pf ppf "Const-SSA export %a has no definition" Const_ssa.Value_id.pp
        value

let add_definition t ~tensor definition =
  let id = Const_ssa.Value_id.of_tensor_id tensor.Tensor_sig.id in
  if Tensor_id.Map.mem tensor.id t.bindings then
    Err.fail (`Binding_overwrite tensor.id)
  else
    let open Err.Syntax in
    let* plan =
      Const_ssa.add t.plan ~id definition
      |> Err.map_error (fun e -> (e :> error))
    in
    Err.return
      { t with plan; bindings = Tensor_id.Map.add tensor.id id t.bindings }

let bind_captured t ~tensor capture =
  add_definition t ~tensor
    (Const_ssa.Leaf { leaf = Captured capture; output = tensor })

let bind_literal t ~tensor payload =
  add_definition t ~tensor
    (Const_ssa.Leaf { leaf = Literal payload; output = tensor })

let bind_apply t ~tensor op =
  add_definition t ~tensor (Const_ssa.Apply { op; output = tensor })

let bind_materialized t ~tensor payload =
  let id = tensor.Tensor_sig.id in
  if Tensor_id.Map.mem id t.materialized then Err.fail (`Binding_overwrite id)
  else
    let open Err.Syntax in
    let* _ = bind_literal empty ~tensor payload in
    Err.return
      { t with materialized = Tensor_id.Map.add id payload t.materialized }

let restrict_and_rename_exports t rename =
  let open Err.Syntax in
  let* bindings =
    Tensor_id.Map.fold
      (fun id value acc ->
        let* acc = acc in
        let* () =
          if Const_ssa.find t.plan value = None then
            Err.fail (`Missing_export value)
          else Err.return ()
        in
        match rename id with
        | None -> Err.return acc
        | Some renamed -> Err.return (Tensor_id.Map.add renamed value acc))
      t.bindings
      (Err.return Tensor_id.Map.empty)
  in
  let materialized =
    Tensor_id.Map.fold
      (fun id value acc ->
        match rename id with
        | None -> acc
        | Some renamed -> Tensor_id.Map.add renamed value acc)
      t.materialized Tensor_id.Map.empty
  in
  Err.return { t with bindings; materialized }

let pp fmt t =
  let pp_binding fmt (tensor, value) =
    Fmt.pf fmt "%a -> %a" Tensor_id.pp tensor Const_ssa.Value_id.pp value
  in
  Fmt.pf fmt "@[<v>exports:@ %a@,plan:@ %a@]"
    Fmt.(list ~sep:cut pp_binding)
    (Tensor_id.Map.bindings t.bindings)
    Const_ssa.pp t.plan
