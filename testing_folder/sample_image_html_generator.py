




output_string = ""

for i in range(10):


	output_string+= "\n"+ '''<div class="sample_image_container">
	    <h3>Image '''+str(i+1)+''':</h3>
	    <p>Image Name:{{msg.payload.image_name['''+str(i)+''']}}</p>
	    <p>Ground Truth Label:{{msg.payload.ground_truth['''+str(i)+''']}}</p>
	    <p>Input Image:</p>
	    <img style="width:25%" src="data:image/jpg;base64, {{msg.payload.input['''+str(i)+''']}}" alt="Task Input Image" />
	</div> '''

print(output_string)