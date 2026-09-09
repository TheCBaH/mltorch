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
  | M2 of { q : bool; k : bool } (* [Wq|1, Wk|1] *)
  | M4 of { d : bool; h : bool; q : bool; k : bool }
  (* [D|1, H|1, Wq|1, Wk|1] *)
  | No_mask

(* Which of query/key/value have their own [batch]/[heads] axis pinned to 1
   rather than the shared real extent -- [Attention.Sdpa.broadcast_batch]
   accepts any per-operand mix of "equal or one side 1" on [N;T;D;H], the
   same rule [Recipe_matmul]'s own [broadcast_side] exercises for its two
   operands. Deliberately excludes "all three pinned": [mask_shape] below
   reads the REAL [c.batch]/[c.heads] value directly (not the per-operand
   one), which is only guaranteed to equal the true broadcast output when at
   least one operand keeps it -- exactly the same reasoning
   [Recipe_matmul]'s [d_bc]/[h_bc] uses to justify reading the shared field
   unconditionally. *)
type broadcast_pin =
  | All_real
  | Q_one
  | K_one
  | V_one
  | Qk_one
  | Qv_one
  | Kv_one

type t = {
  batch : int;
  heads : int;
  sq : int; (* query sequence *)
  sk : int; (* key/value sequence *)
  e : int; (* head dim; the value feature dim is also [e] -- Ev = E, F4 *)
  batch_bc : broadcast_pin;
  heads_bc : broadcast_pin;
  mask : mask_kind;
  scale : float option;
}

(* Correlation is by construction (every shape below is derived from the same
   fields), so there is nothing for [cascade] to repair. *)
let cascade c = c

let pin_extents pin ~real =
  let q1, k1, v1 =
    match pin with
    | All_real -> (false, false, false)
    | Q_one -> (true, false, false)
    | K_one -> (false, true, false)
    | V_one -> (false, false, true)
    | Qk_one -> (true, true, false)
    | Qv_one -> (true, false, true)
    | Kv_one -> (false, true, true)
  in
  let e pinned = if pinned then 1 else real in
  (e q1, e k1, e v1)

let query_shape c =
  let bq, _, _ = pin_extents c.batch_bc ~real:c.batch in
  let hq, _, _ = pin_extents c.heads_bc ~real:c.heads in
  [ bq; hq; c.sq; c.e ]

let key_shape c =
  let _, bk, _ = pin_extents c.batch_bc ~real:c.batch in
  let _, hk, _ = pin_extents c.heads_bc ~real:c.heads in
  [ bk; hk; c.sk; c.e ]

let value_shape c =
  let _, _, bv = pin_extents c.batch_bc ~real:c.batch in
  let _, _, hv = pin_extents c.heads_bc ~real:c.heads in
  [ bv; hv; c.sk; c.e ]

let mask_shape c =
  let axis real broadcast = if broadcast then 1 else real in
  match c.mask with
  | M2 { q; k } -> Some [ axis c.sq q; axis c.sk k ]
  | M4 { d; h; q; k } ->
      Some [ axis c.batch d; axis c.heads h; axis c.sq q; axis c.sk k ]
  | No_mask -> None

let scale c = c.scale

(* Every representable mask kind, COMBINED broadcasts included (op8-impl-
   review.md P2): the single-axis-broadcast list this recipe shipped with
   first covered the full form and each axis broadcast alone, but never two
   or more axes broadcast together (a 2D [1,1] mask, a 4D [D,1,1,Wk] one,
   ...) -- shapes [Attention.Sdpa.output_shape]'s per-axis check accepts
   just as readily, and so needed exercising against the real kernel just as
   much. 1 (No_mask) + 4 (M2) + 16 (M4) = 21 points; each mask_kind's
   booleans still make it flash-admissible by construction, so enumerating
   every combination costs nothing beyond the walk step count. The outer
   sequence remains rank-grouped (no mask, then 2D, then 4D): it determines
   the seeded walk trace and intentionally differs from [mask_kind]'s
   declaration order. *)
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

let axes ~batch ~heads ~sq ~sk ~e ~batch_bc ~heads_bc ~mask ~scale () =
  Walk.
    [
      field_axis "batch" batch (fun c v -> { c with batch = v });
      field_axis "heads" heads (fun c v -> { c with heads = v });
      field_axis "sq" sq (fun c v -> { c with sq = v });
      field_axis "sk" sk (fun c v -> { c with sk = v });
      field_axis "e" e (fun c v -> { c with e = v });
      field_axis "batch_bc" batch_bc (fun c v -> { c with batch_bc = v });
      field_axis "heads_bc" heads_bc (fun c v -> { c with heads_bc = v });
      field_axis "mask" mask (fun c v -> { c with mask = v });
      field_axis "scale" scale (fun c v -> { c with scale = v });
    ]

let pp_mask ppf = function
  | M2 { q; k } -> Fmt.pf ppf "2d(q=%b,k=%b)" q k
  | M4 { d; h; q; k } -> Fmt.pf ppf "4d(d=%b,h=%b,q=%b,k=%b)" d h q k
  | No_mask -> Fmt.string ppf "none"

let pp_pin ppf = function
  | All_real -> Fmt.string ppf "real"
  | Q_one -> Fmt.string ppf "q"
  | K_one -> Fmt.string ppf "k"
  | V_one -> Fmt.string ppf "v"
  | Qk_one -> Fmt.string ppf "qk"
  | Qv_one -> Fmt.string ppf "qv"
  | Kv_one -> Fmt.string ppf "kv"

let pp ppf c =
  Format.fprintf ppf
    "{batch=%d heads=%d sq=%d sk=%d e=%d batch_bc=%a heads_bc=%a mask=%a \
     scale=%s}"
    c.batch c.heads c.sq c.sk c.e pp_pin c.batch_bc pp_pin c.heads_bc pp_mask
    c.mask
    (match c.scale with None -> "default" | Some s -> Printf.sprintf "%g" s)
