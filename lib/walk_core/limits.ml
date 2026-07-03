(* Global caps for the random-walk language, common to every operation. Range
   axes draw within these bounds and shapes are validated against them, so one
   place controls how large the synthesized tensors (and thus the per-step
   runtime) can get. Op walks are functors over [S], so a single shared instance
   parameterises all of them; raise the caps at the call site for heavier
   fuzzing. *)

type t = {
  max_kernel : int;  (** conv/pool kernel extent *)
  max_stride : int;
  max_pad : int;
  max_dilation : int;
  max_channels : int;  (** "filters" / channel count (e.g. 256) *)
  max_extent : int;  (** per spatial axis (H/W) *)
  max_batch : int;  (** N *)
  max_numel : int;  (** total elements of a synthesized shape — bounds runtime *)
}

(* Modest caps: keep the pure Direct-vs-Symbolic tests fast. The mechanism is
   the cap, not the specific number — raise these (channels to 256, …) for
   broader coverage. *)
let default =
  {
    max_kernel = 5;
    max_stride = 3;
    max_pad = 2;
    max_dilation = 3;
    max_channels = 32;
    max_extent = 16;
    max_batch = 2;
    max_numel = 16384;
  }

module type S = sig
  val limits : t
end
