(* Fallback walk recipe for single-tensor ops without hand-written meta: vary the
   one tensor's dim sizes while holding its rank fixed, leaving every scalar arg
   at its declared default. Rank is held constant so any fixed structural arg
   stays in range; a single tensor accepts any shape, so [cascade] is identity.

   Weaker than a family recipe (it never explores scalar params), but valid. *)

type t = { shape : int list }

let cascade c = c
let shape c = c.shape

(* Candidate dim sizes for the one mutating axis. *)
let sizes = [ 2; 3; 4; 6; 8 ]

(* One axis: pick a dim index and a new size for it, keeping the rank fixed. *)
let axes =
  [
    {
      Walk.name = "shape";
      mutate =
        (fun pcg c ->
          let rank = List.length c.shape in
          let idx, pcg = Walk.pick pcg (List.init rank Fun.id) in
          let v, pcg = Walk.pick pcg sizes in
          ( { shape = List.mapi (fun i s -> if i = idx then v else s) c.shape },
            pcg ));
    };
  ]

let pp ppf c =
  Format.fprintf ppf "{shape=[%s]}"
    (String.concat "," (List.map string_of_int c.shape))
