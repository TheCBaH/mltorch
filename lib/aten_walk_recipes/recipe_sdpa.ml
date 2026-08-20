(* The scaled_dot_product_attention.default walk recipe (op8-impl.md commit 4,
   F9). Q/K/V shapes are all DERIVED from one correlated (batch, heads, sq,
   sk, e) tuple -- there is no [ev] field, deliberately: a distinct value
   feature dimension moves ATen's CPU dispatch off the flash kernel (F4), so
   the recipe must be unable to express it at all, the same discipline
   [Recipe_norm] applies to its [normalized] list.

   [mask_kind] is what makes every representable mask FLASH-ADMISSIBLE BY
   CONSTRUCTION: each boolean says "this axis is 1 rather than its real
   extent", so there is no way to build the [Wq,Wk]-shaped-but-not-really
   mask [check_attn_mask_shape] would reject. That check's failures are
   silent (F4) -- ATen falls back to the `math` backend rather than erroring
   -- so an invalid mask drawn by an uncorrelated recipe would not even show
   up as a `skipped` walk line; it would quietly report against the wrong
   oracle. *)

type mask_kind =
  | No_mask
  | M2 of { q : bool; k : bool } (* [Wq|1, Wk|1] *)
  | M4 of { d : bool; h : bool; q : bool; k : bool }
(* [D|1, H|1, Wq|1, Wk|1] *)

type t = {
  batch : int;
  heads : int;
  sq : int; (* query sequence *)
  sk : int; (* key/value sequence *)
  e : int; (* head dim; the value feature dim is also [e] -- Ev = E, F4 *)
  mask : mask_kind;
  scale : float option;
}

(* Correlation is by construction (every shape below is derived from the same
   fields), so there is nothing for [cascade] to repair. *)
let cascade c = c
let query_shape c = [ c.batch; c.heads; c.sq; c.e ]
let key_shape c = [ c.batch; c.heads; c.sk; c.e ]
let value_shape c = key_shape c

let mask_shape c =
  let axis real broadcast = if broadcast then 1 else real in
  match c.mask with
  | No_mask -> None
  | M2 { q; k } -> Some [ axis c.sq q; axis c.sk k ]
  | M4 { d; h; q; k } ->
      Some [ axis c.batch d; axis c.heads h; axis c.sq q; axis c.sk k ]

let scale c = c.scale

(* Every representable mask kind, COMBINED broadcasts included (op8-impl-
   review.md P2): the single-axis-broadcast list this recipe shipped with
   first covered the full form and each axis broadcast alone, but never two
   or more axes broadcast together (a 2D [1,1] mask, a 4D [D,1,1,Wk] one,
   ...) -- shapes [Attention.Sdpa.output_shape]'s per-axis check accepts
   just as readily, and so needed exercising against the real kernel just as
   much. 1 (No_mask) + 4 (M2) + 16 (M4) = 21 points; each mask_kind's
   booleans still make it flash-admissible by construction, so enumerating
   every combination costs nothing beyond the walk step count. *)
let all_mask_kinds =
  let bools = [ false; true ] in
  No_mask
  :: List.concat_map (fun q -> List.map (fun k -> M2 { q; k }) bools) bools
  @ List.concat_map
      (fun d ->
        List.concat_map
          (fun h ->
            List.concat_map
              (fun q -> List.map (fun k -> M4 { d; h; q; k }) bools)
              bools)
          bools)
      bools

let axes ~batch ~heads ~sq ~sk ~e ~mask ~scale () =
  Walk.
    [
      field_axis "batch" batch (fun c v -> { c with batch = v });
      field_axis "heads" heads (fun c v -> { c with heads = v });
      field_axis "sq" sq (fun c v -> { c with sq = v });
      field_axis "sk" sk (fun c v -> { c with sk = v });
      field_axis "e" e (fun c v -> { c with e = v });
      field_axis "mask" mask (fun c v -> { c with mask = v });
      field_axis "scale" scale (fun c v -> { c with scale = v });
    ]

let pp_mask ppf = function
  | No_mask -> Fmt.string ppf "none"
  | M2 { q; k } -> Fmt.pf ppf "2d(q=%b,k=%b)" q k
  | M4 { d; h; q; k } -> Fmt.pf ppf "4d(d=%b,h=%b,q=%b,k=%b)" d h q k

let pp ppf c =
  Format.fprintf ppf "{batch=%d heads=%d sq=%d sk=%d e=%d mask=%a scale=%s}"
    c.batch c.heads c.sq c.sk c.e pp_mask c.mask
    (match c.scale with None -> "default" | Some s -> Printf.sprintf "%g" s)
