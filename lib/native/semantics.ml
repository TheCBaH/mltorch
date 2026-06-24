(* Abstract domain for ops: both [t] (value) and ['role ix] (index) are abstract,
   so one op functor runs as [Direct] (ix=int, t=float), [Symbolic] (ix=iexpr,
   t=expr), or a future [Footprint] (ix=interval). See .ai/native_compute_design.md §1.

   Index phantom roles: [index] is a position known ≥ 0 (what [load] accepts);
   [delta] is a signed affine. Windowed arithmetic is built in [delta] and
   converted to [index] only via [clamp_low] (sound: ≥ 0) or [assume_index]
   (one encapsulated unchecked claim, in Window_axis.window). *)
type index
type delta

module type SEMANTICS = sig
  type t
  type 'role ix

  (* value domain — minimal basis. [max]/[min]/[relu] are NOT here: they derive
     from [select]+[lt] at each call site (relu = select (lt x 0) 0 x), so a
     new activation costs no new primitive, Expr constructor, or eval/pp arm. *)
  val const : float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val exp : t -> t
  val sqrt : t -> t
  (* transcendental: genuinely primitive, not select-expressible (sqrt is the
     rms-norm normaliser; rsqrt is just 1 / sqrt) *)

  (* boolean domain + selection — the scalable basis for activations/clamps.
     [select c a b] is [a] when [c] holds else [b]; e.g.
     [relu x = select (lt x (const 0.)) (const 0.) x]. *)
  type b

  val lt : t -> t -> b
  val select : b -> t -> t -> t

  (* index domain — affine expressions in [delta]; [load] needs [index].
     [clamp_low] (max 0) converts delta→index soundly; [assume_index] is the
     one unchecked cast, encapsulated in Window_axis.window (where clip holds). *)
  val izero : index ix
  val iext : Dim.extent Dim.t -> delta ix
  val iconst : int -> delta ix
  val of_index : index ix -> delta ix
  val iadd : delta ix -> delta ix -> delta ix
  val iscale : int -> delta ix -> delta ix
  val imin : delta ix -> delta ix -> delta ix
  val clamp_low : delta ix -> index ix
  val assume_index : delta ix -> index ix

  type input

  val load : input -> (Axis.t -> index ix) -> t
  val sum : lo:index ix -> hi:delta ix -> (index ix -> t) -> t
  val maxr : lo:index ix -> hi:delta ix -> (index ix -> t) -> t
end
