(* The single global Limits instance shared by every native op walk — the one
   knob bounding kernel/channel/extent sizes and total numel across all ops.
   Raise the caps here (or swap in a different module) for heavier fuzzing. *)

module L : Walk_core.Limits.S = struct
  let limits = Walk_core.Limits.default
end
