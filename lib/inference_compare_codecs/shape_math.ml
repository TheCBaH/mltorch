let fail e = Err.fail ~pp_error:Codec_error.pp e

let element_count extents =
  let rec go acc = function
    | [] -> Ok acc
    | d :: rest ->
        if Int64.compare d 0L < 0 then
          fail (Codec_error.Shape (Negative_extent d))
        else if Int64.equal d 0L then
          (* Zero wins over any overflow later in the list, but later extents
             are still validated as nonnegative. *)
          if List.exists (fun x -> Int64.compare x 0L < 0) rest then
            fail
              (Codec_error.Shape
                 (Negative_extent
                    (List.find (fun x -> Int64.compare x 0L < 0) rest)))
          else Ok 0L
        else if Int64.compare acc (Int64.div Int64.max_int d) > 0 then
          fail Codec_error.Count_overflow
        else go (Int64.mul acc d) rest
  in
  go 1L extents

let byte_count ~width extents =
  let ( let* ) = Err.bind in
  let* n = element_count extents in
  let limit = Int64.of_int Sys.max_string_length in
  let w = Int64.of_int width in
  if Int64.equal n 0L then Ok 0
  else if Int64.compare n (Int64.div limit w) > 0 then
    fail
      (Codec_error.Resource_exceeded
         {
           requested =
             (if Int64.compare n (Int64.div Int64.max_int w) > 0 then
                Int64.max_int
              else Int64.mul n w);
           limit;
         })
  else Ok (Int64.to_int (Int64.mul n w))

let to_ints extents = List.map Int64.to_int extents

let strides dims =
  let rec go = function
    | [] -> ([], 1)
    | d :: rest ->
        let ss, acc = go rest in
        (acc :: ss, acc * max d 1)
  in
  fst (go dims)
