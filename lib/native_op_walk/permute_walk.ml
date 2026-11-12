(* The random-walk config spaces of the ops in [Permute], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Permute = struct
  open Permute.Permute

  (* Walk config: a full 6-axis shape plus a permutation drawn from a fixed
     candidate set of whole valid perms -- op3-impl.md row 3.3b, the first
     Permute walk. Candidates are WEIGHTED AWAY FROM IDENTITY (none of them is
     the identity permutation): a walk that spends its steps there is the
     "axis that exists and is never exercised" test/native/native_walk_test.ml's
     own header warns about. At least one is a 3-cycle -- the only shape of
     permutation that makes DIRECTION observable, since a two-element swap is
     its own inverse and cannot distinguish a reversed mapping from the
     correct one (op3-impl.md commit 6's mutation table). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; perm : perm }

    let candidate_perms : perm list =
      let open Axis in
      [
        (* pairwise swaps *)
        [ (N, N); (T, T); (D, D); (H, W); (W, H); (C, C) ];
        [ (N, N); (T, T); (D, D); (H, H); (W, C); (C, W) ];
        [ (N, N); (T, T); (D, D); (H, C); (W, W); (C, H) ];
        (* 3-cycles over H/W/C, each direction *)
        [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ];
        [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ];
        (* long-range swaps, so the walk is not confined to H/W/C *)
        [ (N, C); (T, T); (D, D); (H, H); (W, W); (C, N) ];
        [ (N, H); (T, T); (D, D); (H, N); (W, W); (C, C) ];
      ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 4 };
        perm = List.hd candidate_perms;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape
    let perm (c : cfg) = c.perm

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "perm" candidate_perms (fun c v -> { c with perm = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a perm=%a}" Walk_core.Shape.pp c.shape pp_perm
        c.perm
  end
end
