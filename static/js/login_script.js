const container = document.getElementById('container');
const registerBtn = document.getElementById('register');
const loginBtn = document.getElementById('login');

function switch_right(){
    registerBtn.addEventListener('click', () => {
        container.classList.add("active");
    });
}

function switch_left(){
    loginBtn.addEventListener('click', () => {
        container.classList.remove("active");
    });
}