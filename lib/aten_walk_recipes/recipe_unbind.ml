(* Unbind-family walk recipe: a rank-4 tensor split along one dim.

   The generated default walk holds [dim] at its schema default of 0 and varies
   only the shape, so the whole negative- and nonzero-dim space is unreachable
   from it. This exists to reach it.

   Rank and dim are correlated by construction rather than by filtering: the
   rank is fixed at 4 and the candidate set is exactly the dims valid for a
   rank-4 tensor, so no mutation can step into a combination ATen rejects — the
   rule in .ai/native_add_op.md about candidate lists of whole valid
   configurations. [cascade] is therefore the identity. *)

type t = { n : int; c : int; h : int; w : int; dim : int }

(* Both spellings of every axis. This positive-then-negative sequence is the
   seeded walk trace's stable coverage order. A negative dim normalises to the same axis as
   its non-negative twin, which is the property the bridge's own normalisation
   has to preserve — and the reason both spellings are walked rather than just
   one of each pair. *)
let all_dims = [ 0; 1; 2; 3; -1; -2; -3; -4 ]
let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]
let dim c = c.dim

let axes ~n ~c ~h ~w ~dim =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "dim" dim (fun c v -> { c with dim = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] dim=%d}" c.n c.c c.h c.w c.dim
