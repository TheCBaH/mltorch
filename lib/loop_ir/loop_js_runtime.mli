(** The named helpers generated code calls, as [Js_ast] built with [Js_build].
    Each is a transcription of its OCaml counterpart, tested directly against it
    ([make loop.js.runtest]), so the emitter references a helper by name instead
    of open-coding a semantic that two backends could then disagree on. *)

(** Closed and alphabetical: the order the prelude is emitted in, so it is
    deterministic. *)
module Name : sig
  type t =
    | Bf16_to_float
    | Coord_failure
    | Erf
    | F16_to_float
    | Float_max
    | I64_from_float_failure
    | Pool_better

  val all : t list
  val to_string : t -> string
end

module Helper : sig
  type t = {
    name : Name.t;
    defines : Js_ident.t list;
        (** every top-level name the body declares: the helper is needed when
            the program uses one of them *)
    body : Js_ast.Stmt.t list;
  }
end

val helper : Name.t -> Helper.t

val helpers : Helper.t list
(** Every helper, in [Name.all] order. *)

val source : string
(** Every helper printed, for a script that calls them all. *)

(** {1 Typed calls}

    The one place each helper's kinds are stated. *)

val bf16_to_float : Js_build.bits16 Js_build.t -> Js_build.num Js_build.t

val coord_failure :
  buffer:Js_build.idx Js_build.t ->
  extents:Js_build.idx Js_build.t list ->
  coord:Js_build.idx Js_build.t list ->
  Js_ast.expr

val erf : Js_build.num Js_build.t -> Js_build.num Js_build.t
val f16_to_float : Js_build.bits16 Js_build.t -> Js_build.num Js_build.t

val float_max :
  Js_build.num Js_build.t -> Js_build.num Js_build.t -> Js_build.num Js_build.t

val i64_from_float_failure : Js_build.num Js_build.t -> Js_ast.expr

val pool_better :
  Js_build.num Js_build.t ->
  Js_build.num Js_build.t ->
  Js_build.bool_ Js_build.t
