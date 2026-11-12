(* The random-walk config spaces of the ops in [Attention], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Sdpa = struct
  open Attention.Sdpa

  (* Config space for [lib/native_op_walk]'s Direct-vs-Symbolic fuzz walk
     NOT the ATen-oracle recipe ([Recipe_sdpa]), which additionally has to
     keep every point on the flash kernel. This walk has no ATen oracle to
     stay on, so its only
     obligation is a VALID Native graph: [batch]/[heads]/[wq]/[wk]/[e] are
     independent fields, but every shape [build] derives is computed FROM
     them (never five independently-drawn [Vec6.shape]s), so a mismatched
     query/key/value/mask shape is unrepresentable by construction and
     [cascade] has nothing to repair. *)
  module Walk (_ : Walk_core.Limits.S) = struct
    type mask_cfg = Absent | Present

    type cfg = {
      batch : int;
      heads : int;
      wq : int;
      wk : int;
      e : int;
      mask : mask_cfg;
      scale : Scale.t;
    }

    let initial =
      {
        batch = 1;
        heads = 2;
        wq = 3;
        wk = 4;
        e = 5;
        mask = Present;
        scale = Scale.Default;
      }

    let cascade c = c

    let query_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:c.batch ~h:c.heads ~w:c.wq ~c:c.e

    let key_shape (c : cfg) =
      Vec6.shape ~n:1 ~t:1 ~d:c.batch ~h:c.heads ~w:c.wk ~c:c.e

    let mask_shape (c : cfg) =
      score_shape ~query_shape:(query_shape c) ~key_shape:(key_shape c)

    let mask_present (c : cfg) = c.mask = Present
    let params (c : cfg) : params = { scale = c.scale }

    let axes =
      Walk_core.Walk.
        [
          field_axis "batch" [ 1; 2 ] (fun c v -> { c with batch = v });
          field_axis "heads" [ 1; 2; 3 ] (fun c v -> { c with heads = v });
          field_axis "wq" [ 1; 2; 3; 5 ] (fun c v -> { c with wq = v });
          field_axis "wk" [ 1; 2; 4; 6 ] (fun c v -> { c with wk = v });
          field_axis "e" [ 1; 3; 5 ] (fun c v -> { c with e = v });
          field_axis "mask" [ Absent; Present ] (fun c v -> { c with mask = v });
          field_axis "scale"
            [ Scale.Default; Scale.Explicit 0.5; Scale.Explicit 1.0 ]
            (fun c v -> { c with scale = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{batch=%d heads=%d wq=%d wk=%d e=%d mask=%s scale=%a}"
        c.batch c.heads c.wq c.wk c.e
        (match c.mask with Absent -> "absent" | Present -> "present")
        pp_scale c.scale
  end
end
