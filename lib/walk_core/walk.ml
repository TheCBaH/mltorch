(* The shared random-walk machinery: the axis abstraction and the loop that,
   given an op's config space (axes + cascade), mutates a config and hands the
   built subject to a backend [verify] for evaluation + comparison.

   Backend-neutral by construction: an op provides its own walk properties (the
   [Op] signature); the *evaluator and comparison* are passed to [run] as
   [verify]. So this lib links neither the ATen C++ build nor the native engine
   — only the operations and the runners that consume them do. *)

type 'cfg axis = { name : string; mutate : Pcg.t -> 'cfg -> 'cfg * Pcg.t }
(** A configurable dimension of an op's parameter space. [mutate] picks a new
    value freely (one [Pcg.next] step); cross-parameter validity is restored
    afterwards by the op's [cascade], not by filtering here. *)

(** The walk properties an operation describes about itself. Lives WITH the op
    (per backend); never shared across backends, since their semantics and valid
    ranges can differ. *)
module type Op_space = sig
  type cfg

  val initial : cfg
  val axes : cfg axis list
  val cascade : cfg -> cfg
  val pp : Format.formatter -> cfg -> unit
end

(** A full walk subject the runner consumes: an [Op_space] plus how to build the
    backend [subject] (an ATen [Op_spec.t], or a native graph + inputs). *)
module type Op = sig
  include Op_space

  type subject

  val target : string
  val build : Pcg.t -> cfg -> subject * Pcg.t
end

(* Reduce a raw [Pcg.next] draw to an index in [0, m). The remainder is taken in
   [Int64] because the draw spans [0, 2^32), which is not a non-negative [int] on
   a 32-bit-[int] runtime such as js_of_ocaml; the result is < m, so narrowing it
   afterwards is always safe. Identical to the former [n mod m] natively. *)
let index_of pcg_out ~m = Int64.to_int (Int64.rem pcg_out (Int64.of_int m))

(* Pick one element from a non-empty list using one [Pcg.next] step. *)
let pick pcg lst =
  let n, pcg = Pcg.next pcg in
  (List.nth lst (index_of n ~m:(List.length lst)), pcg)

(* A field axis: pick a candidate value with the PCG and set it via [setter].
   No validity filtering -- the engine applies [cascade] after the mutation.
   The enumerated form, for genuinely non-numeric axes (padding mode, dim
   subsets, bool); numeric dimensions should use the range/compound axes below. *)
let field_axis name candidates setter =
  {
    name;
    mutate =
      (fun pcg cfg ->
        let v, pcg = pick pcg candidates in
        (setter cfg v, pcg));
  }

(* Draw an int in [lo, hi] from the PCG (one step). *)
let int_in pcg ~lo ~hi =
  let hi = max lo hi in
  let n, pcg = Pcg.next pcg in
  (lo + index_of n ~m:(hi - lo + 1), pcg)

(* A numeric axis over a whole range [lo, hi] -- any int, not an enumerated set.
   The natural lower bound is the value's domain (1 for kernel/stride, 0 for
   pad); the upper bound comes from the global Limits. *)
let int_axis name ~lo ~hi setter =
  {
    name;
    mutate =
      (fun pcg cfg ->
        let v, pcg = int_in pcg ~lo ~hi in
        (setter cfg v, pcg));
  }

type hw = { h : int; w : int }
(** A compound H/W value (kernel/stride/pad/dilation) carried as one entry. *)

(* A compound H/W axis: draws h and w each in [lo, hi], setting both fields in
   one step via [setter]. *)
let hw_axis name ~lo ~hi setter =
  {
    name;
    mutate =
      (fun pcg cfg ->
        let h, pcg = int_in pcg ~lo ~hi in
        let w, pcg = int_in pcg ~lo ~hi in
        (setter cfg { h; w }, pcg));
  }

(* A tensor-shape axis: pick one mutable axis of the [Shape.t] entry and resize
   it to an int within that axis's cap, clamped to the numel budget ([Shape.resize]
   keeps the shape valid). One compound entry replaces per-dimension scalar axes. *)
let shape_axis name (limits : Limits.t) ~get ~set =
  let choices : Shape.axis list = [ `N; `H; `W; `C ] in
  {
    name;
    mutate =
      (fun pcg cfg ->
        let shape = get cfg in
        let axis, pcg = pick pcg choices in
        let v, pcg = int_in pcg ~lo:1 ~hi:(Shape.cap limits axis) in
        (set cfg (Shape.resize limits shape ~axis v), pcg));
  }

(* Drive the walk for one op. Step 0 is the cascaded initial config; each step
   picks an axis, mutates it freely, re-cascades to restore validity, builds the
   subject, and hands it to [verify]. Stops on the first [verify] = false so the
   failing config is the last line printed. *)
let run (type s) (module M : Op with type subject = s)
    ~(verify : Format.formatter -> s -> bool) ~ppf ~pcg ~steps =
  let run_step cfg pcg =
    let subject, pcg = M.build pcg cfg in
    (verify ppf subject, pcg)
  in
  let initial = M.cascade M.initial in
  Format.fprintf ppf "step 0: %a@." M.pp initial;
  let ok, pcg = run_step initial pcg in
  if ok then
    let rec loop i cfg pcg =
      if i > steps then ()
      else
        let axis, pcg = pick pcg M.axes in
        let cfg, pcg = axis.mutate pcg cfg in
        let cfg = M.cascade cfg in
        Format.fprintf ppf "step %d [%s]: %a@." i axis.name M.pp cfg;
        let ok, pcg = run_step cfg pcg in
        if ok then loop (i + 1) cfg pcg
    in
    loop 1 initial pcg
