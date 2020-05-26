import random
import re
from itertools import combinations

def mapper(key, value):
    # key: None
    # value: one line of input file
    # --------------------------------------------------------------------------
    # Define the number of hash functions.
    numHashes = 975
    # Record the maximum shingle ID.
    maxShingleID = 8192
    # The next largest prime number above 'maxShingleID'.
    nextPrime = 8209

    # Random hash function will take the form of: h(x) = (a*x + b) % c
    # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
    # a prime number just greater than maxShingleID (i.e., nextPrime).

    random.seed(0)
    def pickRandomCoeffs(k):
      # Create a list of 'k' random values.
      randList = []

      while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        while randIndex in randList:
          randIndex = random.randint(0, maxShingleID)

        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

      return randList

    # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)


    # Get the shingle set.
    shingleIDSet = [int(m) for m in value.split()[1:]]

    # The resulting minhash signature for this page.
    signature = []

    # For each of the random hash functions...
    for i in range(0, numHashes):
        # For each of the shingles actually in the page, calculate its hash code
        # using hash function 'i'.

        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
        # the maximum possible value output by the hash.
        minHashCode = nextPrime + 1

        # For each shingle in the page...
        for shingleID in shingleIDSet:
          # Evaluate the hash function.
          hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

          # Track the lowest hash code seen.
          if hashCode < minHashCode:
            minHashCode = hashCode

        # Add the smallest hash code value as component number 'i' of the signature.
        signature.append(minHashCode)
        print("signature========= ")
        print(signature)

    # Define the number of hash functions per band.
    num_per_band = 25

    # Get signatures for each band.
    band_signature = []
    for j in range(0, numHashes, num_per_band):
        band_signature.append(signature[j : j+num_per_band])
    
    print("band_signature===========")
    print(band_signature)
    # Generate a bucket (a set of bandID and band).
    for band_id, band in enumerate(band_signature):
        bucket = [band_id, band]
        bucket = ' '.join(map(str, bucket))
        print("bucket============")
        print(bucket)

        # Extract page numbers from the original pageID.
        if value.split()[0] == "PAGE_000000000":
            page_id = 0
        else: page_id = re.search(r'[1-9][0-9]*', value.split()[0]).group()
        yield bucket, page_id


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # --------------------------------------------------------------------------
    if len(values) >= 2:
        comb = list(combinations(values, 2))
        yield int(sorted(comb[0],key=int)[0]), int(sorted(comb[0],key=int)[1])
