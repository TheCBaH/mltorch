(* A scalar literal: an op's Scalar argument, or one explicit tensor value.

   Canonical form is a single-key object so the int/float/bool tag is explicit:
   {"int": 1} | {"float": 1.0} | {"float": "0x80000000"} | {"bool": true}.
   As a convenience a bare JSON bool decodes to [Bool] and a bare number to
   [Float] (rounded to f32) — but a bare number can't be an [Int] (JSON has one
   number type), so use the object form when an integer scalar is required. *)

type t = Bool of bool | Float of Float32.t | Int of int

let jsont : t Jsont.t =
  Jsont.map ~kind:"scalar"
    ~dec:(fun json ->
      match json with
      | Jsont.Bool (b, _) -> Bool b
      | Jsont.Number (n, _) -> Float (Float32.to_f32 n)
      | Jsont.Object _ ->
          Spec_util.union ~kind:"scalar"
            [
              ("bool", fun v -> Bool (Spec_util.dec Jsont.bool v));
              ("float", fun v -> Float (Spec_util.dec Float32.jsont v));
              ("int", fun v -> Int (Spec_util.dec Jsont.int v));
            ]
            json
      | _ ->
          Jsont.Error.msgf Jsont.Meta.none
            "scalar: expected a number, bool, or single-key object")
    ~enc:(function
      | Bool b -> Spec_util.single ~case:"bool" (Spec_util.jbool b)
      | Float f ->
          Spec_util.single ~case:"float" (Spec_util.enc Float32.jsont f)
      | Int i -> Spec_util.single ~case:"int" (Spec_util.jint i))
    Jsont.json
