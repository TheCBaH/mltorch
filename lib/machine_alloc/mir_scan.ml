(* Linear scan proper, one bank at a time, for [Mir_linear_scan]: an
   interval's pieces, the registers fixed constraints block, and the
   free-until and eviction decisions that place or spill each piece. *)

open Machine_ir
module Loc = Mir_phys.Loc

module type POOL = sig
  include Mir_ref_alloc.REGISTERS

  val view : Mir_target.Bank.t -> bits:int -> int -> Mir_target.View.t
  (** register [k] of a bank at a width *)

  val allocatable : Mir_target.Bank.t -> int list
  (** the registers allocation may assign, in preference order *)
end

(* Fault injection for the evidence suite: one deliberate allocation defect
   each, which the checker or the interpreter must catch. No consumer passes
   one. *)
module Mutation = struct
  type t =
    | Call_interval  (** a call's clobbered views not blocked *)
    | Hole  (** an inactive interval's later ranges ignored *)
    | Split_move  (** a split inside a block without its transition move *)
end

module Make (T : Mir_sel.TARGET) (R : POOL) = struct
  module S = Mir_sel.Make (T)
  module V = Mir_phys_verify.Make (T)
  module Lv = Mir_liveness.Make (T)

  type func = (S.Stage.op, S.Stage.term) Mir_func.t
  type block = (S.Stage.op, S.Stage.term) Mir_block.t

  let is_flags (v : Mir_value.t) = Mir_type.equal v.Mir_value.ty Mir_type.Flags

  let shape (v : Mir_value.t) =
    match V.shape v.Mir_value.ty with
    | Some s -> s
    | None -> invalid_arg "Mir_linear_scan: a value with no shape"

  module Where = struct
    type t = In of int | Spilled | Unassigned
  end

  (* One allocated piece of a value's interval. *)
  module Piece = struct
    type t = {
      value : Mir_value.t;
      mutable ranges : (int * int) list;  (** half-open, ascending *)
      mutable uses : int list;  (** ascending *)
      mutable where : Where.t;
    }

    let start p = fst (List.hd p.ranges)
    let stop p = snd (List.hd (List.rev p.ranges))
    let covers p x = List.exists (fun (a, b) -> a <= x && x < b) p.ranges

    (* The first position at or after [x] both pieces cover. *)
    let intersection p q ~from =
      List.fold_left
        (fun acc (a, b) ->
          List.fold_left
            (fun acc (c, d) ->
              let lo = max (max a c) from and hi = min b d in
              if lo < hi then
                match acc with Some m when m <= lo -> acc | _ -> Some lo
              else acc)
            acc q.ranges)
        None p.ranges

    let next_use p ~from = List.find_opt (fun u -> u >= from) p.uses

    (* Splits at [s] (even): [p] keeps what lies before, the result holds the
       rest, unassigned. *)
    let split p s =
      let before, after =
        List.fold_right
          (fun (a, b) (bf, af) ->
            if b <= s then ((a, b) :: bf, af)
            else if a >= s then (bf, (a, b) :: af)
            else ((a, s) :: bf, (s, b) :: af))
          p.ranges ([], [])
      in
      let q =
        {
          value = p.value;
          ranges = after;
          uses = List.filter (fun u -> u >= s) p.uses;
          where = Where.Unassigned;
        }
      in
      p.ranges <- before;
      p.uses <- List.filter (fun u -> u < s) p.uses;
      q
  end

  (* A register blocked over [lo, hi) by an instruction's constraint. *)
  module Fixed = struct
    type t = {
      view : Mir_target.View.t;
      lo : int;
      hi : int;
      own : Mir_value.t list;
          (** the instruction's results, written after it: never blocked *)
    }
  end

  (* The even position a split for a use at [u] happens at: before the
     instruction reading it, or before the last instruction of a block whose
     terminator reads it (the block's start, if it has none). *)
  let split_for u = if u land 1 = 0 then u else u - 1

  type st = { mutation : Mutation.t option; mutable next_block : int }

  let mutated st m = st.mutation = Some m

  (* --- linear scan, one bank at a time ----------------------------------- *)

  let scan st ~bank ~fixed ~depth (pieces : Piece.t list) =
    let pool = R.allocatable bank in
    let view_of (p : Piece.t) k =
      R.view bank ~bits:(snd (shape p.Piece.value)) k
    in
    let weight u = 10. ** float_of_int (depth u) in
    let all = ref [] in
    let unhandled =
      ref
        (List.sort (fun a b -> compare (Piece.start a) (Piece.start b)) pieces)
    in
    let push p =
      if p.Piece.ranges <> [] then
        unhandled :=
          List.merge
            (fun a b -> compare (Piece.start a) (Piece.start b))
            [ p ] !unhandled
    in
    let active = ref [] and inactive = ref [] in
    let reg_of (p : Piece.t) =
      match p.Piece.where with Where.In k -> Some k | _ -> None
    in
    (* the first position a fixed interval on [k] blocks [cur], bit for bit *)
    let fixed_block cur k =
      List.fold_left
        (fun acc (f : Fixed.t) ->
          if
            Mir_target.View.overlap f.Fixed.view (view_of cur k)
            && (not (List.exists (Mir_value.equal cur.Piece.value) f.Fixed.own))
            && not
                 (mutated st Mutation.Call_interval
                 && f.Fixed.own <> []
                 && f.Fixed.lo land 1 = 1)
          then
            match
              Piece.intersection cur
                { cur with Piece.ranges = [ (f.Fixed.lo, f.Fixed.hi) ] }
                ~from:(Piece.start cur)
            with
            | Some x -> (
                match acc with Some m when m <= x -> acc | _ -> Some x)
            | None -> acc
          else acc)
        None fixed
    in
    (* a spilled stretch from [p]'s start, then a fresh attempt before its next
       use after [after] *)
    let spill_from (p : Piece.t) ~after =
      p.Piece.where <- Where.Spilled;
      all := p :: !all;
      match
        List.find_opt
          (fun u -> split_for u > after && split_for u > Piece.start p)
          p.Piece.uses
      with
      | Some u ->
          let q = Piece.split p (split_for u) in
          push q
      | None -> ()
    in
    let rec loop () =
      match !unhandled with
      | [] -> ()
      | cur :: rest ->
          unhandled := rest;
          let pos = Piece.start cur in
          (* expire and reclassify *)
          let still, ended =
            List.partition (fun p -> Piece.stop p > pos) !active
          in
          ignore ended;
          let act, inact = List.partition (fun p -> Piece.covers p pos) still in
          let still_i = List.filter (fun p -> Piece.stop p > pos) !inactive in
          let act2, inact2 =
            List.partition (fun p -> Piece.covers p pos) still_i
          in
          active := act @ act2;
          inactive := inact @ inact2;
          let free_until k =
            if List.exists (fun p -> reg_of p = Some k) !active then pos
            else
              let i =
                if mutated st Mutation.Hole then None
                else
                  List.fold_left
                    (fun acc p ->
                      if reg_of p = Some k then
                        match Piece.intersection cur p ~from:pos with
                        | Some x -> (
                            match acc with
                            | Some m when m <= x -> acc
                            | _ -> Some x)
                        | None -> acc
                      else acc)
                    None !inactive
              in
              let f = fixed_block cur k in
              match (i, f) with
              | Some a, Some b -> min a b
              | Some a, None | None, Some a -> a
              | None, None -> max_int
          in
          let hint =
            List.find_map
              (fun (p : Piece.t) ->
                if Mir_value.equal p.Piece.value cur.Piece.value then reg_of p
                else None)
              !all
          in
          let best =
            List.fold_left
              (fun acc k ->
                let f = free_until k in
                match acc with
                | Some (_, g) when g > f -> acc
                | Some (bk, g) when g = f && Some bk = hint -> acc
                | Some (_, g) when g = f && Some k <> hint -> acc
                | _ -> Some (k, f))
              None pool
          in
          let assign k =
            cur.Piece.where <- Where.In k;
            all := cur :: !all;
            active := cur :: !active
          in
          (match best with
          | Some (k, f) when f >= Piece.stop cur -> assign k
          | Some (k, f) when split_for f > pos && f > pos ->
              (* free for a while: split before it is taken *)
              let s = split_for f in
              let q = Piece.split cur s in
              assign k;
              push q
          | _ -> (
              (* blocked: evict the farthest next use, or spill [cur] *)
              let score k =
                let uses =
                  List.filter_map
                    (fun p ->
                      if
                        reg_of p = Some k
                        && (Piece.covers p pos
                           || Option.is_some
                                (Piece.intersection cur p ~from:pos))
                      then
                        Some
                          (Option.value ~default:max_int
                             (Piece.next_use p ~from:pos))
                      else None)
                    (!active @ !inactive)
                in
                match fixed_block cur k with
                | Some x when x <= pos + 1 -> None
                | _ ->
                    let u = List.fold_left min max_int uses in
                    if u = max_int then Some infinity
                    else Some (float_of_int (u - pos) /. weight u)
              in
              let candidate =
                List.fold_left
                  (fun acc k ->
                    match (score k, acc) with
                    | None, _ -> acc
                    | Some s, Some (_, t) when t >= s -> acc
                    | Some s, _ -> Some (k, s))
                  None pool
              in
              let own =
                match Piece.next_use cur ~from:pos with
                | Some u -> float_of_int (u - pos) /. weight u
                | None -> infinity
              in
              match candidate with
              | Some (k, s) when s >= own ->
                  (* evict every occupant of [k] that meets [cur] *)
                  let s_e = split_for pos in
                  let evict (p : Piece.t) =
                    if Piece.covers p pos then (
                      active := List.filter (fun q -> q != p) !active;
                      if s_e > Piece.start p then
                        let r = Piece.split p s_e in
                        spill_from r ~after:pos
                      else (
                        all := List.filter (fun q -> q != p) !all;
                        spill_from p ~after:pos))
                    else
                      match Piece.intersection cur p ~from:pos with
                      | Some _ ->
                          inactive := List.filter (fun q -> q != p) !inactive;
                          (* split where it is live again: a block start *)
                          let y =
                            List.fold_left
                              (fun acc (a, _) ->
                                if a > pos && acc = None then Some a else acc)
                              None p.Piece.ranges
                          in
                          Option.iter
                            (fun y ->
                              let r = Piece.split p (split_for y) in
                              push r)
                            y
                      | None -> ()
                  in
                  List.iter
                    (fun p -> if reg_of p = Some k then evict p)
                    (!active @ !inactive);
                  (* still blocked later by a fixed interval: split before it *)
                  (match fixed_block cur k with
                  | Some x when split_for x > pos ->
                      let q = Piece.split cur (split_for x) in
                      push q
                  | _ -> ());
                  assign k
              | _ -> spill_from cur ~after:pos));
          loop ()
    in
    loop ();
    !all
end
