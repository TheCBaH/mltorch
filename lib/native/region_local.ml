module Shape = struct
  type t = Scalar

  let scalar = Scalar
  let pp fmt Scalar = Fmt.string fmt "scalar"
end

type t = { id : Expr.Local_var.t; shape : Shape.t; value : Expr.Value.t }

let scalar ~id ~value = { id; shape = Shape.scalar; value }
