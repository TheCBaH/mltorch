(* _native_batch_norm_legit_no_training.default walk recipe: a rank-4 [n;c;h;w]
   input, running_mean/running_var pinned to shape [c] (the channel extent),
   and independently optional weight/bias of the same shape -- the same
   four-affine-state shape as [Recipe_norm], reused here because the bridge
   materialises an absent weight/bias as ones/zeros of exactly [c] rather than
   reshaping anything (op_bridge.ml, the
   [_native_batch_norm_legit_no_training.default] arm).

   [momentum] is not a walk axis: the bridge's decode for this op never reads
   it (eval-mode batch norm doesn't touch the running stats), so a value
   distinct from the schema default would exercise nothing -- unlike
   [Recipe_norm]'s [cudnn], which IS read and explicitly discarded, and is
   walked to prove that. *)

type t = {
  n : int;
  c : int;
  h : int;
  w : int;
  eps : float;
  weight : bool;
  bias : bool;
}

let cascade c = c
let input_shape c = [ c.n; c.c; c.h; c.w ]
let channel_shape c = [ c.c ]
let has_weight c = c.weight
let has_bias c = c.bias
let eps c = c.eps
let momentum = 0.1

let axes ~n ~c ~h ~w ~eps ~weight ~bias =
  Walk.
    [
      field_axis "n" n (fun t v -> { t with n = v });
      field_axis "c" c (fun t v -> { t with c = v });
      field_axis "h" h (fun (t : t) v -> { t with h = v });
      field_axis "w" w (fun (t : t) v -> { t with w = v });
      field_axis "eps" eps (fun t v -> { t with eps = v });
      field_axis "weight" weight (fun t v -> { t with weight = v });
      field_axis "bias" bias (fun t v -> { t with bias = v });
    ]

let pp ppf t =
  Format.fprintf ppf "{input=[%d,%d,%d,%d] eps=%g weight=%b bias=%b}" t.n t.c
    t.h t.w t.eps t.weight t.bias
