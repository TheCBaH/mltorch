(* See monad.mli. *)

module State = struct
  type ('s, 'a) t = 's -> 'a * 's

  let return x s = (x, s)

  let bind m f s =
    let x, s = m s in
    f x s

  let ( let* ) = bind

  let map m f s =
    let x, s = m s in
    (f x, s)

  let ( let+ ) = map
  let get s = (s, s)
end
