(* See wide_permute.mli. *)

type op = Clone | Permute of Permute.Permute.perm | Reshape of Vec6.shape

type step =
  | Reshape4 of Shape4.t
  | Permute4 of (Axis4.t * Axis4.t) list * Shape4.t

let max_blocks = 6
let max_permutes = 3

exception Abort

(* Every product is a sub-product of one tensor's extents, hence at most its
   element count, which graph construction bounds inside a 32-bit [int]. *)
let product = List.fold_left ( * ) 1
let ext shape axis = Dim.to_int (Vec6.get shape axis)

type state = {
  mutable next : int;
  mutable extents : (int * int) list;  (** atom id -> extent *)
  mutable base : int list;  (** the source's atoms, in flat order *)
  mutable groups : (Axis.t * int list) list;
      (** the current tensor's non-unit axes, each with its atoms *)
}

let extent st a = List.assoc a st.extents

(* Cut atom [a] into a major part of extent [first] and the rest, everywhere it
   is held. A cut is the same in every tensor of the run because an atom's
   pieces keep their order wherever it sits. *)
let split st a first =
  let e = extent st a in
  let a1 = st.next and a2 = st.next + 1 in
  st.next <- st.next + 2;
  st.extents <- (a1, first) :: (a2, e / first) :: st.extents;
  let sub l =
    List.concat_map (fun b -> if b = a then [ a1; a2 ] else [ b ]) l
  in
  st.base <- sub st.base;
  st.groups <- List.map (fun (ax, l) -> (ax, sub l)) st.groups;
  (a1, a2)

let flat st = List.concat_map snd st.groups

let reshape st target =
  let rec take need atoms acc =
    if need = 1 then (List.rev acc, atoms)
    else
      match atoms with
      | [] -> raise Abort
      | a :: rest ->
          let e = extent st a in
          if need mod e = 0 then take (need / e) rest (a :: acc)
          else if e mod need = 0 then
            let a1, a2 = split st a need in
            take 1 (a2 :: rest) (a1 :: acc)
          else raise Abort
  in
  let groups, rest =
    List.fold_left
      (fun (groups, atoms) axis ->
        let need = ext target axis in
        if need = 1 then (groups, atoms)
        else
          let taken, atoms = take need atoms [] in
          ((axis, taken) :: groups, atoms))
      ([], flat st)
      Axis.all
  in
  if rest <> [] then raise Abort;
  st.groups <- List.rev groups

let permute st perm =
  st.groups <-
    List.filter_map
      (fun out ->
        Option.map
          (fun g -> (out, g))
          (List.assoc_opt (Permute.Permute.lookup perm out) st.groups))
      Axis.all

(* ---- blocks --------------------------------------------------------------- *)

(* Maximal runs of source-adjacent atoms that are also adjacent, in the same
   order, in the result. Returns each block's extent (source order) and the
   result as a sequence of block indices. *)
let blocks st =
  let cur = flat st in
  let rec follows a = function
    | x :: (y :: _ as rest) -> if x = a then Some y else follows a rest
    | _ -> None
  in
  let cuts =
    List.mapi
      (fun i a ->
        match List.nth_opt st.base (i + 1) with
        | Some b -> follows a cur <> Some b
        | None -> true)
      st.base
  in
  let block_of, _ =
    List.fold_left2
      (fun (acc, k) a cut -> ((a, k) :: acc, if cut then k + 1 else k))
      ([], 0) st.base cuts
  in
  let n = List.fold_left (fun m (_, k) -> max m (k + 1)) 0 block_of in
  let sizes =
    List.init n (fun k ->
        product
          (List.filter_map
             (fun (a, j) -> if j = k then Some (extent st a) else None)
             block_of))
  in
  let order =
    List.fold_left
      (fun acc a ->
        let k = List.assoc a block_of in
        match acc with j :: _ when j = k -> acc | _ -> k :: acc)
      [] cur
    |> List.rev
  in
  (sizes, order)

(* ---- search over block orders --------------------------------------------- *)

(* Every way to cut [l] into between 2 and 4 contiguous runs. *)
let rec cuts_into k l =
  match (k, l) with
  | 1, _ :: _ -> [ [ l ] ]
  | _, [] | _, [ _ ] -> []
  | _ ->
      List.concat_map
        (fun i ->
          let head = List.filteri (fun j _ -> j <= i) l
          and tail = List.filteri (fun j _ -> j > i) l in
          List.map (fun runs -> head :: runs) (cuts_into (k - 1) tail))
        (List.init (List.length l - 1) Fun.id)

let rec permutations = function
  | [] -> [ [] ]
  | l ->
      List.concat_map
        (fun x ->
          List.map
            (fun p -> x :: p)
            (permutations (List.filter (fun y -> y <> x) l)))
        l

type move = { runs : int list list; order : int list }

let moves arrangement =
  List.concat_map
    (fun k ->
      List.concat_map
        (fun runs ->
          List.filter_map
            (fun order ->
              if order = List.init k Fun.id then None else Some { runs; order })
            (permutations (List.init k Fun.id)))
        (cuts_into k arrangement))
    [ 2; 3; 4 ]

let apply { runs; order } = List.concat_map (fun j -> List.nth runs j) order

(* Shortest sequence of moves from [start] to [goal], at most [max_permutes]. *)
let search ~start ~goal =
  let seen = Hashtbl.create 64 in
  Hashtbl.replace seen start ();
  let rec level frontier depth =
    match List.find_opt (fun (a, _) -> a = goal) frontier with
    | Some (_, path) -> Some (List.rev path)
    | None ->
        if depth = max_permutes then None
        else
          let next =
            List.concat_map
              (fun (a, path) ->
                List.filter_map
                  (fun m ->
                    let a' = apply m in
                    if Hashtbl.mem seen a' then None
                    else begin
                      Hashtbl.replace seen a' ();
                      Some (a', (a, m) :: path)
                    end)
                  (moves a))
              frontier
          in
          if next = [] then None else level next (depth + 1)
  in
  level [ (start, []) ] 0

(* ---- emission ------------------------------------------------------------- *)

let shape_of_extents es =
  let labels = List.filteri (fun i _ -> i >= 4 - List.length es) Axis4.all in
  List.fold_left2
    (fun s label e -> Shape4.set s label (Dim.extent e))
    (Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:1)
    labels es

let steps_of ~sizes ~x ~y path =
  let size b = List.nth sizes b in
  let cur = ref x in
  let out = ref [] in
  let reshape_to shape =
    if not (Shape4.equal !cur shape) then begin
      out := Reshape4 shape :: !out;
      cur := shape
    end
  in
  List.iter
    (fun (_, { runs; order }) ->
      let run_size r = product (List.map size r) in
      reshape_to (shape_of_extents (List.map run_size runs));
      let k = List.length runs in
      let labels = List.filteri (fun i _ -> i >= 4 - k) Axis4.all in
      let perm =
        List.map
          (fun a ->
            match
              List.find_opt
                (fun (_, l) -> Axis4.equal a l)
                (List.mapi (fun j l -> (j, l)) labels)
            with
            | Some (j, _) -> (a, List.nth labels (List.nth order j))
            | None -> (a, a))
          Axis4.all
      in
      let shape =
        shape_of_extents (List.map (fun j -> run_size (List.nth runs j)) order)
      in
      out := Permute4 (perm, shape) :: !out;
      cur := shape)
    path;
  reshape_to y;
  List.rev !out

let plan ?source ~x ~y ops =
  try
    let x6 = Option.value source ~default:(Shape4.to_vec6 x) in
    let st = { next = 0; extents = []; base = []; groups = [] } in
    List.iter
      (fun axis ->
        let e = ext x6 axis in
        if e > 1 then begin
          let a = st.next in
          st.next <- a + 1;
          st.extents <- (a, e) :: st.extents;
          st.base <- st.base @ [ a ];
          st.groups <- st.groups @ [ (axis, [ a ]) ]
        end)
      Axis.all;
    List.iter
      (function
        | Clone -> ()
        | Permute perm -> permute st perm
        | Reshape target -> reshape st target)
      ops;
    let sizes, order = blocks st in
    if List.length sizes > max_blocks then raise Abort;
    let start = List.init (List.length sizes) Fun.id in
    match search ~start ~goal:order with
    | None -> None
    | Some path -> Some (steps_of ~sizes ~x ~y path)
  with Abort -> None
