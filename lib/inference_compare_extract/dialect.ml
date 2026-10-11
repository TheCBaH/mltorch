open Inference_compare_schema

module type S = sig
  type graph
  type body

  val dialect : Id.Dialect.t
  val describe : graph -> Graph_desc.t
  val extract : graph -> Extraction.t -> (body, Extract_error.t) Err.t
  val describe_body : body -> Graph_desc.t
  val validate : body -> (unit, Extract_error.t) Err.t
end

let realize (type g b) (module D : S with type graph = g and type body = b)
    (g : g) (plan : Extraction.t) =
  let ( let* ) = Err.bind in
  let* body = D.extract g plan in
  let* () = D.validate body in
  let* () = Reconcile.check plan (D.describe_body body) in
  Ok body
