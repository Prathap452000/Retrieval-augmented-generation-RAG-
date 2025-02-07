def classify_sampling_case(f_s, f_max):
    """
    Classifies the relationship between sampling frequency (f_s) and max frequency (f_max).
    Provides feedback on the aliasing behavior.

    Parameters:
    - f_s: Sampling frequency (in Hz)
    - f_max: Maximum frequency of the signal (in Hz)
    """

    # Calculate Nyquist frequency
    f_N = f_s / 2

    # Case 1: Properly sampled (f_s > 2 * f_max)
    if f_s > 2 * f_max:
        print(f"\nCase 1: Properly Sampled")
        print(f"Sampling Frequency (f_s) = {f_s} Hz is more than twice the Maximum Frequency (f_max) = {f_max} Hz.")
        print("The signal is properly sampled, and there is no aliasing.")
    
    # Case 2: Exact Nyquist rate (f_s == 2 * f_max)
    elif f_s == 2 * f_max:
        print(f"\nCase 2: Nyquist Rate")
        print(f"Sampling Frequency (f_s) = {f_s} Hz is exactly twice the Maximum Frequency (f_max) = {f_max} Hz.")
        print("This is the threshold sampling rate, where the highest frequency can be captured, but any increase in frequency may cause aliasing.")
    
    # Case 3: Aliasing will occur (f_s < 2 * f_max)
    elif f_s < 2 * f_max:
        print(f"\nCase 3: Aliasing Occurs")
        print(f"Sampling Frequency (f_s) = {f_s} Hz is less than twice the Maximum Frequency (f_max) = {f_max} Hz.")
        print("Aliasing will occur as the signal is under-sampled and high frequencies will fold back into the lower frequencies.")
        
    # Provide Nyquist frequency for reference
    print(f"Nyquist Frequency: {f_N} Hz")
 
# Main function to interact with the user
def main():
    print("Sampling Frequency Classifier!") 

    # Taking input from the user
    try:
        f_s = float(input("Enter the Sampling Frequency (f_s) in Hz: "))
        f_max = float(input("Enter the Maximum Frequency (f_max) in Hz: "))
        
        # Classify the case based on the input
        classify_sampling_case(f_s, f_max)
    
    except ValueError:
        print("Please enter valid numeric values for frequencies.")

# Run the main function
if __name__ == "__main__":
    main()

